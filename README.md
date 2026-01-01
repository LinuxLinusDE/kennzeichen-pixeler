# Plater – Pixelate License Plates in 4K Videos (Apple Silicon)

This project detects license plates in 4K videos and pixelates them automatically. Optimized for Apple Silicon (M4) with MPS and CPU fallback. Privacy first: it's better to pixelate too much than to miss a plate.

## Requirements
- Python 3.10+
- ffmpeg via Homebrew: `brew install ffmpeg`
- A YOLOv8 license plate model as a `.pt` file (e.g. `best.pt`)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Quick start (simple)
1) Open a terminal in the project folder.
2) Activate the virtual environment:
```bash
source .venv/bin/activate
```
3) Run (adjust filenames):
```bash
python blur_plates_m4.py --input input.mp4 --output output.mp4 --weights best.pt
```

## Examples
HEVC default (4K, MPS, audio is preserved):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output.mp4 \
  --weights /path/to/plate_model.pt
```

H.264 compatible output (plays everywhere):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output_h264.mp4 \
  --weights /path/to/plate_model.pt \
  --codec h264 \
  --bitrate 20M
```

Force software encoding (if hardware encoding fails):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output_sw.mp4 \
  --weights /path/to/plate_model.pt \
  --force_sw
```

Quick test with lower detection width (faster, less accurate):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output_fast.mp4 \
  --weights /path/to/plate_model.pt \
  --work_w 1280
```

## Key parameters
- `work_w`: detection width (e.g. 1280 or 1920). 0 = original resolution.
- `imgsz`: YOLO inference size.
- `conf`: confidence threshold (higher = fewer detections).
- `blocks`: pixel block size (smaller = coarser pixelation).
- `pad`: safety padding around each box (pixels).
- `no_pixel_zone`: no-pixel zone as `x1,x2,y1,y2` in percent (default `0,22,59,100` for bottom-left HUD).
- `force_sw`: force software encoding.

## Simple usage steps
1) Put your video (e.g. `input.mp4`) and weights (e.g. `best.pt`) into the project folder.
2) Open a terminal in the project folder.
3) Run the command from Quick start.
4) The result will be saved as `output.mp4` in the same folder.

## Insta360 note
Recommendation: reframe/flat export to 16:9 in Insta360 Studio first, then pixelate.

## Troubleshooting
VideoToolbox error -12908 (HW encoding fails): often caused by pixel format negotiation. The script forces `nv12` for VideoToolbox. You can test:
```bash
ffmpeg -y -f lavfi -i testsrc2=size=3840x2160:rate=60 -t 2 -vf format=nv12 -c:v hevc_videotoolbox -b:v 12M vt_test.mp4
```

Rotation looks wrong: some files have rotation metadata. Normalize via ffmpeg:
```bash
ffmpeg -i input.mp4 -vf "transpose=0" -c:a copy normalized.mp4
```

Variable framerate: normalize first:
```bash
ffmpeg -i input.mp4 -vsync cfr -r 25 -c:v libx264 -c:a copy normalized.mp4
```

## Privacy note
Goal is unreadability. Use coarse pixels (`blocks` small) and sufficient `pad` so nothing is missed.
