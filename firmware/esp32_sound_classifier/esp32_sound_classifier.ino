#include <WiFi.h>
#include <HTTPClient.h>
#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

#include <esp_timer.h>
#include <math.h>
#include <string.h>

// Set these for your Wi-Fi network and HTTP endpoint.
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* SERVER_URL = "http://10.104.47.161:8000/predict";
const char* DEVICE_ID = "esp32-1";

constexpr int kMicPin = 34;  // ADC1 pin; MAX module output must stay within 0-3.3 V.
constexpr int kSampleRate = 16000;
constexpr int kFrameSamples = 500;
constexpr int kFrameCount = 32;
constexpr int kFeatureBins[] = {2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 224, 240};
constexpr int kFeaturesPerFrame = sizeof(kFeatureBins) / sizeof(kFeatureBins[0]);
constexpr int kArenaSize = 24 * 1024;

alignas(16) uint8_t tensor_arena[kArenaSize];
float feature_input[kFeatureCount];
tflite::MicroMutableOpResolver<3> resolver;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;

bool connect_wifi() {
  if (WiFi.status() == WL_CONNECTED) {
    return true;
  }
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  const unsigned long started = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - started < 20000) {
    delay(250);
    Serial.print(".");
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("Wi-Fi connection failed; will retry later.");
    return false;
  }
  Serial.print("Wi-Fi connected, IP: ");
  Serial.println(WiFi.localIP());
  return true;
}

bool add_audio_features(float* output) {
  float samples[kFrameSamples];
  const float window_energy = []() {
    float total = 0.0f;
    for (int n = 0; n < kFrameSamples; ++n) {
      const float value = 0.5f - 0.5f * cosf(
          2.0f * PI * n / (kFrameSamples - 1));
      total += value * value;
    }
    return total;
  }();

  for (int frame_index = 0; frame_index < kFrameCount; ++frame_index) {
    const int64_t frame_start_us = esp_timer_get_time();
    double mean = 0.0;
    for (int n = 0; n < kFrameSamples; ++n) {
      const int64_t target_us = frame_start_us
          + (static_cast<int64_t>(n) * 1000000) / kSampleRate;
      while (esp_timer_get_time() < target_us) {
      }
      samples[n] = (static_cast<float>(analogRead(kMicPin)) - 2048.0f) / 2048.0f;
      mean += samples[n];
    }
    mean /= kFrameSamples;

    for (int n = 0; n < kFrameSamples; ++n) {
      const float window = 0.5f - 0.5f * cosf(
          2.0f * PI * n / (kFrameSamples - 1));
      samples[n] = (samples[n] - static_cast<float>(mean)) * window;
    }

    for (int feature_index = 0; feature_index < kFeaturesPerFrame; ++feature_index) {
      const int bin = kFeatureBins[feature_index];
      const float coefficient = 2.0f * cosf(2.0f * PI * bin / kFrameSamples);
      float previous = 0.0f;
      float previous_previous = 0.0f;
      for (int n = 0; n < kFrameSamples; ++n) {
        const float current = samples[n] + coefficient * previous - previous_previous;
        previous_previous = previous;
        previous = current;
      }
      const float power = previous * previous
          + previous_previous * previous_previous
          - coefficient * previous * previous_previous;
      const float safe_power = fmaxf(power / window_energy, 1e-12f);
      const int offset = frame_index * kFeaturesPerFrame + feature_index;
      output[offset] = (log10f(safe_power) - g_feature_mean[offset])
          / g_feature_scale[offset];
    }
  }
  return true;
}

void send_prediction(const char* label, int class_id, float confidence,
                     const float* scores) {
  if (!connect_wifi()) {
    return;
  }
  WiFiClient client;
  HTTPClient http;
  if (!http.begin(client, SERVER_URL)) {
    Serial.println("Could not initialize HTTP request.");
    return;
  }
  http.addHeader("Content-Type", "application/json");
  String body = "{\"device_id\":\"";
  body.reserve(512);
  body += DEVICE_ID;
  body += "\",\"label\":\"";
  body += label;
  body += "\",\"class_id\":";
  body += String(class_id);
  body += ",\"confidence\":";
  body += String(confidence, 5);
  body += ",\"scores\":{";
  for (int i = 0; i < kClassCount; ++i) {
    if (i > 0) {
      body += ",";
    }
    body += "\"";
    body += g_class_names[i];
    body += "\":";
    body += String(scores[i], 5);
  }
  body += "}}";

  const int status = http.POST(body);
  if (status > 0) {
    Serial.printf("Server returned HTTP %d: %s\n", status,
                  http.getString().c_str());
  } else {
    Serial.printf("HTTP POST failed: %s\n", http.errorToString(status).c_str());
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  analogSetPinAttenuation(kMicPin, ADC_11db);

  const tflite::Model* model = tflite::GetModel(g_sound_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("TFLite model schema does not match the installed runtime.");
    while (true) {
      delay(1000);
    }
  }
  if (resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddRelu() != kTfLiteOk ||
      resolver.AddSoftmax() != kTfLiteOk) {
    Serial.println("Could not register required TFLite operators.");
    while (true) {
      delay(1000);
    }
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kArenaSize);
  interpreter = &static_interpreter;
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("Tensor allocation failed with the 24 KB arena.");
    while (true) {
      delay(1000);
    }
  }
  model_input = interpreter->input(0);
  model_output = interpreter->output(0);
  if (model_input->type != kTfLiteFloat32 ||
      model_input->bytes != kFeatureCount * sizeof(float) ||
      model_output->type != kTfLiteFloat32 ||
      model_output->bytes != kClassCount * sizeof(float)) {
    Serial.println("Model tensor shape/type does not match the generated header.");
    while (true) {
      delay(1000);
    }
  }
  connect_wifi();
  Serial.println("Sound classifier ready.");
}

void loop() {
  if (!add_audio_features(feature_input)) {
    return;
  }
  memcpy(model_input->data.f, feature_input, sizeof(feature_input));
  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("TFLite inference failed.");
    delay(100);
    return;
  }

  int best_index = 0;
  float best_score = model_output->data.f[0];
  for (int i = 1; i < kClassCount; ++i) {
    if (model_output->data.f[i] > best_score) {
      best_score = model_output->data.f[i];
      best_index = i;
    }
  }
  const int reported_class_id = best_index;
  const char* label = g_class_names[best_index];
  Serial.printf("Prediction: %s (%.3f)\n", label, best_score);
  send_prediction(label, reported_class_id, best_score, model_output->data.f);
}
