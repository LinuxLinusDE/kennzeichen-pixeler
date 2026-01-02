#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import time
from typing import Dict, Tuple

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except Exception as e:
    print("Fehlende Abhaengigkeiten. Bitte Umgebung einrichten und Pakete installieren:")
    print("  python3 -m venv .venv")
    print("  source .venv/bin/activate")
    print("  pip install -U pip")
    print("  pip install -r requirements.txt")
    print("")
    print("Danach starten mit:")
    print("  python blur_plates_m4.py --input input.mp4 --output output.mp4 --weights best.pt")
    raise SystemExit(2) from e


def apply_pad(x1: int, y1: int, x2: int, y2: int, pad: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


def pixelate_roi(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, blocks: int) -> None:
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    if rh == 0 or rw == 0:
        return
    blocks = max(1, int(blocks))
    small_w = min(blocks, rw)
    small_h = max(1, int(rh * (small_w / float(rw))))
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    pixel = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
    img[y1:y2, x1:x2] = pixel


def boxes_overlap(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def build_ffmpeg_cmd(
    output: str,
    input_path: str,
    w: int,
    h: int,
    fps: float,
    codec: str,
    bitrate: str,
    use_sw: bool,
) -> list:
    if codec == "hevc":
        vcodec = "libx265" if use_sw else "hevc_videotoolbox"
    elif codec == "h264":
        vcodec = "libx264" if use_sw else "h264_videotoolbox"
    else:
        raise ValueError(f"Unsupported codec: {codec}")

    pix_fmt = "yuv420p" if use_sw else "nv12"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        f"{fps}",
        "-i",
        "pipe:0",
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-vf",
        f"format={pix_fmt}",
        "-pix_fmt",
        pix_fmt,
        "-c:v",
        vcodec,
        "-b:v",
        bitrate,
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        output,
    ]
    if not use_sw:
        idx = cmd.index("-b:v")
        cmd[idx:idx] = ["-allow_sw", "1"]
        if codec == "h264":
            cmd[idx:idx] = ["-profile:v", "high", "-level:v", "5.2"]
    return cmd


def open_video(input_path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Video nicht oeffnbar: {input_path}")
    return cap


def get_fps(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1.0 or fps > 240.0:
        return 25.0
    return fps


def probe_bitrate(input_path: str) -> str:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "default=nk=1:nw=1",
            input_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if out.isdigit():
            return f"{int(out) // 1000}k"
    except Exception:
        pass
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=bit_rate",
            "-of",
            "default=nk=1:nw=1",
            input_path,
        ]
        out = subprocess.check_output(cmd, text=True).strip()
        if out.isdigit():
            return f"{int(out) // 1000}k"
    except Exception:
        pass
    return "50M"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kennzeichen in Videos erkennen und verpixeln (Apple Silicon/MPS).")
    p.add_argument("--input", help="Input-Video (MP4)")
    p.add_argument("--output", help="Output-Video (MP4)")
    p.add_argument("--weights", help="YOLOv8 .pt weights file")
    p.add_argument("--device", default="mps", help="Ultralytics device, z.B. mps oder cpu")
    p.add_argument("--conf", type=float, default=None, help="Confidence Threshold")
    p.add_argument("--imgsz", type=int, default=None, help="YOLO imgsz")
    p.add_argument("--work_w", type=int, default=None, help="Arbeitsbreite fuer Detektion (0 = Original)")
    p.add_argument("--blocks", type=int, default=None, help="Pixel-Blockgroesse; kleiner = grober")
    p.add_argument("--pad", type=int, default=None, help="Sicherheitsrand in Pixel")
    p.add_argument("--codec", choices=["hevc", "h264"], default="hevc", help="Video codec")
    p.add_argument("--bitrate", default=None, help="Video bitrate, z.B. 50M oder auto")
    p.add_argument(
        "--preset",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Preset fuer Speed/Qualitaet",
    )
    p.add_argument("--force_sw", action="store_true", help="Software-Encoding erzwingen (libx265/libx264)")
    p.add_argument("--debug_overlay", action="store_true", help="BBox-Overlay fuer Debug einzeichnen")
    p.add_argument("--test_minutes", type=int, default=0, help="Nur die ersten N Minuten verarbeiten (0 = alles)")
    p.add_argument("--log_every", type=int, default=200, help="Log alle n Frames")
    p.add_argument(
        "--no_pixel_zone",
        default="0,22,59,100",
        help="No-Pixel-Zone in Prozent als x1,x2,y1,y2 (z.B. 0,22,59,100)",
    )
    p.add_argument(
        "--no_pixel_zone2",
        default="78,100,59,100",
        help="Zweite No-Pixel-Zone in Prozent (z.B. 78,100,59,100)",
    )
    return p.parse_args()


def prompt_value(label: str) -> str:
    return input(f"{label}: ").strip()


def resolve_paths(args: argparse.Namespace) -> None:
    if args.input and args.output and args.weights:
        return
    if not sys.stdin.isatty():
        return
    print("Interaktiver Modus: Bitte fehlende Werte eingeben.")
    if not args.input:
        args.input = prompt_value("Input-Video (z.B. input.mp4)")
    if not args.output:
        args.output = prompt_value("Output-Video (z.B. output.mp4)")
    if not args.weights:
        args.weights = prompt_value("Weights-Datei (z.B. best.pt)")


def auto_weights_path() -> str:
    if os.path.isfile("best.pt"):
        return "best.pt"
    candidates = [f for f in os.listdir(".") if f.endswith(".pt") and os.path.isfile(f)]
    if not candidates:
        return ""
    candidates.sort()
    return candidates[0]


def apply_preset(args: argparse.Namespace) -> None:
    presets: Dict[str, Dict[str, object]] = {
        "fast": {"conf": 0.3, "imgsz": 960, "work_w": 1280, "blocks": 16, "pad": 20, "bitrate": "auto"},
        "balanced": {"conf": 0.25, "imgsz": 1280, "work_w": 1920, "blocks": 16, "pad": 20, "bitrate": "auto"},
        "quality": {"conf": 0.2, "imgsz": 1600, "work_w": 0, "blocks": 16, "pad": 24, "bitrate": "auto"},
    }
    preset = presets.get(args.preset, presets["balanced"])
    if args.conf is None:
        args.conf = float(preset["conf"])
    if args.imgsz is None:
        args.imgsz = int(preset["imgsz"])
    if args.work_w is None:
        args.work_w = int(preset["work_w"])
    if args.blocks is None:
        args.blocks = int(preset["blocks"])
    if args.pad is None:
        args.pad = int(preset["pad"])
    if args.bitrate is None:
        args.bitrate = str(preset["bitrate"])


def main() -> int:
    if len(sys.argv) == 1:
        print("Plater - Kennzeichen verpixeln (einfacher Start)")
        print("Vorbereitung (einmalig):")
        print("  python3 -m venv .venv")
        print("  source .venv/bin/activate")
        print("  pip install -U pip")
        print("  pip install -r requirements.txt")
        print("Beispiel:")
        print("  python blur_plates_m4.py --input input.mp4 --output output.mp4 --weights best.pt")
        print("Kurz-Erklaerung:")
        print("  Erkennt Kennzeichen im Video und verpixelt sie fuer Datenschutz.")
        print("Wichtige Optionen (kurz):")
        print("  --codec hevc|h264     (Standard: hevc)")
        print("  --preset fast|balanced|quality")
        print("  --bitrate auto        (passt Bitrate an das Original an)")
        print("  --work_w 1920         (schneller, etwas weniger genau)")
        print("  --imgsz 1280          (bessere Erkennung, langsamer)")
        print("  --conf 0.25           (niedriger = mehr Treffer)")
        print("  --blocks 16           (Pixelstaerke, kleiner = grober)")
        print("  --pad 20              (Sicherheitsrand)")
        print("  --no_pixel_zone 0,22,59,100  (HUD-Bereich aussparen)")
        print("  --no_pixel_zone2 78,100,59,100 (HUD-Bereich rechts aussparen)")
        print("  --test_minutes 2      (nur erste 2 Minuten verarbeiten)")
        print("  --debug_overlay       (BBox-Overlay fuer Debug)")
        print("  --force_sw            (Software-Encoding erzwingen)")
        print("Weitere Hilfe:")
        print("  python blur_plates_m4.py -h")
        return 0
    args = parse_args()
    resolve_paths(args)
    if not args.weights:
        args.weights = auto_weights_path()
    if not args.weights:
        print("Weights nicht gefunden. Bitte --weights angeben.", file=sys.stderr)
        return 2
    apply_preset(args)
    exit_code = 0

    if not args.input or not args.output:
        print("Input/Output fehlt. Bitte --input und --output angeben.", file=sys.stderr)
        return 2

    if not os.path.isfile(args.weights):
        print(f"Weights nicht gefunden: {args.weights}", file=sys.stderr)
        return 2

    if not shutil_which("ffmpeg"):
        print("ffmpeg nicht gefunden. Bitte installieren: brew install ffmpeg", file=sys.stderr)
        return 2
    if (args.bitrate or "").lower() == "auto" and not shutil_which("ffprobe"):
        print("ffprobe nicht gefunden. Bitte ffmpeg komplett installieren: brew install ffmpeg", file=sys.stderr)
        return 2

    try:
        cap = open_video(args.input)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    fps = get_fps(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if w <= 0 or h <= 0:
        print("Konnte Videoauflosung nicht ermitteln.", file=sys.stderr)
        cap.release()
        return 2

    try:
        model = YOLO(args.weights)
    except Exception as e:
        print(f"YOLO Modell konnte nicht geladen werden: {e}", file=sys.stderr)
        cap.release()
        return 2

    use_sw = args.force_sw or os.environ.get("FORCE_SW", "") == "1"
    bitrate = args.bitrate
    if isinstance(bitrate, str) and bitrate.lower() == "auto":
        bitrate = probe_bitrate(args.input)
    cmd = build_ffmpeg_cmd(args.output, args.input, w, h, fps, args.codec, bitrate, use_sw)

    proc = None
    frame_idx = 0
    max_frames = 0
    if args.test_minutes and args.test_minutes > 0:
        max_frames = int(fps * 60 * args.test_minutes)
    if max_frames and total_frames > 0:
        total_frames = min(total_frames, max_frames)
    start_time = time.time()
    zones = []
    for zone_arg in [args.no_pixel_zone, args.no_pixel_zone2]:
        try:
            zx1p, zx2p, zy1p, zy2p = [float(v) for v in zone_arg.split(",")]
            zones.append((zx1p, zy1p, zx2p, zy2p))
        except Exception:
            continue
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            det_frame = frame
            scale = 1.0
            if args.work_w and args.work_w > 0 and args.work_w < w:
                scale = args.work_w / float(w)
                new_h = int(h * scale)
                det_frame = cv2.resize(frame, (args.work_w, new_h), interpolation=cv2.INTER_AREA)

            nz_list = []
            for zx1p, zy1p, zx2p, zy2p in zones:
                nz_list.append(
                    (
                        int(w * (zx1p / 100.0)),
                        int(h * (zy1p / 100.0)),
                        int(w * (zx2p / 100.0)),
                        int(h * (zy2p / 100.0)),
                    )
                )

            try:
                results = model.predict(
                    det_frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    device=args.device,
                    verbose=False,
                )
            except Exception as e:
                if args.device != "cpu":
                    results = model.predict(
                        det_frame,
                        conf=args.conf,
                        imgsz=args.imgsz,
                        device="cpu",
                        verbose=False,
                    )
                else:
                    raise e

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    xyxy = b.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy.tolist()
                    if scale != 1.0:
                        x1 = int(x1 / scale)
                        y1 = int(y1 / scale)
                        x2 = int(x2 / scale)
                        y2 = int(y2 / scale)
                    x1, y1, x2, y2 = apply_pad(x1, y1, x2, y2, args.pad, w, h)
                    if nz_list and any(boxes_overlap((x1, y1, x2, y2), nz) for nz in nz_list):
                        continue
                    pixelate_roi(frame, x1, y1, x2, y2, args.blocks)
                    if args.debug_overlay:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if proc.stdin is None:
                raise RuntimeError("ffmpeg stdin nicht verfuegbar")
            proc.stdin.write(frame.tobytes())

            frame_idx += 1
            if args.log_every > 0 and frame_idx % args.log_every == 0:
                elapsed = time.time() - start_time
                fps_eff = frame_idx / elapsed if elapsed > 0 else 0.0
                if total_frames > 0 and fps_eff > 0:
                    remaining = max(total_frames - frame_idx, 0)
                    eta_sec = int(remaining / fps_eff)
                    eta_min = eta_sec // 60
                    eta_rem = eta_sec % 60
                    pct = (frame_idx / total_frames) * 100.0
                    print(f"Processed frames: {frame_idx} | {pct:.1f}% | ETA {eta_min}m {eta_rem}s")
                else:
                    print(f"Processed frames: {frame_idx}")
            if max_frames and frame_idx >= max_frames:
                break

    except BrokenPipeError:
        print("ffmpeg Pipe abgebrochen", file=sys.stderr)
        exit_code = 3
    except Exception as e:
        print(f"Fehler: {e}", file=sys.stderr)
        exit_code = 3
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if proc and proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
        if proc:
            try:
                ret = proc.wait()
                if ret != 0:
                    print(f"ffmpeg exit code: {ret}", file=sys.stderr)
                    exit_code = 3
            except Exception:
                pass

    return exit_code


def shutil_which(cmd: str) -> str:
    try:
        import shutil
        return shutil.which(cmd)
    except Exception:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())
