# UrbanSound8K ESP32 sound classifier

This project extracts matching audio features on the PC and ESP32, trains a
small classifier on UrbanSound8K, exports a float32 TensorFlow Lite model, and
posts one prediction per one-second microphone window to an HTTP endpoint.

The ten output classes are UrbanSound8K's classes. The ESP32 always reports
the class with the highest softmax score, even when that score is low. There
is no `unknown` class or confidence threshold; treat low scores as uncertain
best-match predictions.

## 1. Install Python dependencies

Use Python 3.10 in a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The repository intentionally excludes the `.venv` environment, the UrbanSound8K
source dataset, and the generated feature cache. Download the dataset locally
and regenerate features by following the next section. The trained TFLite model
and ESP32 header are small and included.

## 2. Prepare UrbanSound8K and train

Download and extract UrbanSound8K so the folder contains `audio/fold1` through
`audio/fold10` and `metadata/UrbanSound8K.csv`. From this project folder run:

```powershell
python extract_features.py --dataset "C:\datasets\UrbanSound8K"
python train_model.py
python convert_to_header.py
```

Feature extraction creates one-second windows with a half-second hop, and
zero-pads clips shorter than one second. Training uses folds 1-8, fold 9 for
validation, and fold 10 for the held-out test score. The generated files are:

- `artifacts/urbansound8k_features.npz`
- `artifacts/sound_classifier.tflite`
- `artifacts/metadata.json`
- `firmware/esp32_sound_classifier/model_data.h`

Training uses a compact 64/32-unit dense network with dropout, early stopping
on fold 9, and reports both fold 10 accuracy and macro F1. With the feature
archive used for this project, this configuration scored 43.5% accuracy on
fold 10 versus 39.4% for the original 48/24-unit model. Results can vary with
the dataset windows and runtime; use the newly generated `metadata.json` score
to see the result on your data.

Extraction prints progress every 100 clips. It uses a vectorized FFT on the
desktop for the same frequency-bin power features calculated by Goertzel on
the ESP32. You can stop extraction with Ctrl+C and rerun the command, but it
will restart from the beginning.

The loader supports PCM and compressed WAV formats, including DVI/IMA ADPCM.
The feature vector contains 32 successive 500-sample frames. Each frame has
12 Hann-windowed Goertzel log-power values. The Arduino implementation uses
the same frame size, frequency bins, window, and per-feature normalization as
the Python pipeline.

## 3. Configure and upload the ESP32

1. Install the ESP32 board support package and the `Chirale_TensorFlowLite`
   Arduino library. The sketch includes `Chirale_TensorFlowLite.h`; do not
   replace it with the older `TensorFlowLite_ESP32` library include.
2. Open `firmware/esp32_sound_classifier/esp32_sound_classifier.ino`.
3. Set `WIFI_SSID`, `WIFI_PASSWORD`, and `SERVER_URL` near the top of the
   sketch. For the included receiver, start it on the computer with
   `python receive_predictions.py`, find that computer's Wi-Fi IPv4 address
   with `ipconfig`, then set `SERVER_URL` to
   `http://<computer-ipv4>:8000/predict` (the computer's current Wi-Fi IPv4
   address is `10.104.47.161` in this setup). If DHCP changes that address,
   update it again in the sketch. Do not use `localhost` or
   `127.0.0.1` in the ESP32 URL: those addresses refer to the ESP32 itself.
   Allow Python through Windows Firewall on the Private network when prompted.
4. Connect the microphone module's analog output to GPIO 34, ground to ground,
   and power it according to its module specifications. Keep the signal at the
   ESP32 ADC pin within 0-3.3 V; never connect a negative or 5 V signal.
5. Select the ESP32 DevKit V1 board and upload. Open Serial Monitor at 115200
   baud to inspect predictions and HTTP responses.

GPIO 34 is an input-only ADC1 pin, so it can be used while Wi-Fi is active.

The ESP32 sends JSON like:

```json
{
  "device_id": "esp32-1",
  "label": "siren",
  "class_id": 8,
  "confidence": 0.82341
}
```

`label` and `class_id` identify the highest-scoring one of the ten classes.
`confidence` is that class's score, and `scores` contains scores for all
classes.
Update the URL and the server's route/payload handling to match your deployment.
This sketch uses plain HTTP; use a TLS-capable client and certificate validation
if predictions must be sent over HTTPS.

`receive_predictions.py` is a minimal development receiver and dashboard host.
Open `http://<computer-ipv4>:8000/` in a browser to view the live confidence
bars. The ESP32 must be running the updated sketch, which posts every class
score along with the top prediction. The receiver prints incoming predictions
and serves the latest scores to the dashboard.
Do not open `dashboard.html` directly from the filesystem; browser requests
must go through the receiver so `/api/predictions` can be reached.
Keep the computer and ESP32 on the same reachable network; guest Wi-Fi or
client-isolation settings can block device-to-device connections. Stop the
receiver with Ctrl+C when finished.

## Notes

- Audio is sampled at 16 kHz, in non-overlapping one-second windows on-device.
- The analog microphone signal is DC-centered per 500-sample frame before
  feature calculation. Use a consistent microphone gain and placement; large
  differences from the training recordings can reduce accuracy.
- The ESP32 always returns a best-match class. A low score indicates an
  uncertain prediction; it does not mean the sound belongs to that class with
  high certainty.
- The model is float32 for simpler conversion and inference. The sketch uses a
  24 KB tensor arena and computes features one 500-sample frame at a time to
  avoid reserving a full second of audio in RAM. If tensor allocation fails on
  your library build, the arena may need to be increased; larger arenas consume
  the ESP32's limited internal RAM.
