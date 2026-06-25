#!/usr/bin/env bash
# Install everything Argus needs to run live on this laptop.
# MediaPipe 0.10.35 + onnxruntime run on this project's Python 3.13 venv, so NO separate
# environment is needed. This installs the runtime deps and downloads the model files.
set -euo pipefail
cd "$(dirname "$0")/.."

VENV="./venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

if [ ! -x "$PY" ]; then
  echo "Creating venv..."
  python3 -m venv "$VENV"
fi

echo "=== Installing runtime dependencies ==="
$PIP install -q --upgrade pip
# core (already present) + live backends + OSC art bridge
$PIP install -q numpy scipy pytest opencv-python pyxdf neurokit2 \
    mediapipe onnxruntime python-osc bleak || {
  echo "NOTE: if mediapipe failed, your Python may be too new; see README."; }

echo "=== Verifying backends import ==="
$PY - <<'PYEOF'
for m in ("cv2","mediapipe","onnxruntime","bleak","neurokit2","pyxdf"):
    try:
        __import__(m); print(f"  ok  {m}")
    except Exception as e:
        print(f"  MISSING {m}: {e}")
PYEOF

echo "=== Downloading MediaPipe models (unlock HR, HRV, respiration, fidget, blink) ==="
mkdir -p models
dl() {  # url dest
  if [ -f "$2" ]; then echo "  have $2"; else
    echo "  fetching $2"; curl -fsSL -o "$2" "$1" || echo "  FAILED: $1"
  fi
}
dl "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" models/face_landmarker.task
dl "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task" models/pose_landmarker.task

echo "=== Optional ONNX models (gaze / affect / action-units) ==="
echo "  These do NOT have stable single-file URLs and may need adapter tuning."
echo "  Affect (HSEmotion): pip package 'hsemotion' ships weights; or export enet_b0_8_va_mtl to ONNX."
echo "  Gaze (L2CS-Net):  PyTorch weights -> export to ONNX (no official ONNX release)."
echo "  Action Units (LibreFace): grab the ONNX from the LibreFace releases."
echo "  Once you have a file, pass it: --affect-model PATH / --gaze-model PATH / --au-model PATH"

echo ""
echo "=== Done. Run it: ==="
echo "  PYTHONPATH=src $PY scripts/run_live.py"
echo "  (auto-uses models/*.task -> HR, HRV, respiration, fidget, blink, all from the camera)"
