#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Kennzeichen in Videos erkennen und verpixeln (Apple Silicon/MPS).")
    p.add_argument("--input", required=True, help="Input-Video (MP4)")
    p.add_argument("--output", required=True, help="Output-Video (MP4)")
    p.add_argument("--weights", required=True, help="YOLOv8 .pt weights file")
    p.add_argument("--device", default="mps", help="Ultralytics device, z.B. mps oder cpu")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence Threshold")
    p.add_argument("--imgsz", type=int, default=960, help="YOLO imgsz")
    p.add_argument("--work_w", type=int, default=1280, help="Arbeitsbreite fuer Detektion (0 = Original)")
    p.add_argument("--blocks", type=int, default=16, help="Pixel-Blockgroesse; kleiner = grober")
    p.add_argument("--pad", type=int, default=20, help="Sicherheitsrand in Pixel")
    p.add_argument("--codec", choices=["hevc", "h264"], default="hevc", help="Video codec")
    p.add_argument("--bitrate", default="12M", help="Video bitrate, z.B. 12M")
    p.add_argument("--force_sw", action="store_true", help="Software-Encoding erzwingen (libx265/libx264)")
    p.add_argument("--log_every", type=int, default=200, help="Log alle n Frames")
    p.add_argument(
        "--no_pixel_zone",
        default="0,22,59,100",
        help="No-Pixel-Zone in Prozent als x1,x2,y1,y2 (z.B. 0,22,59,100)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    exit_code = 0

    if not os.path.isfile(args.weights):
        print(f"Weights nicht gefunden: {args.weights}", file=sys.stderr)
        return 2

    if not shutil_which("ffmpeg"):
        print("ffmpeg nicht gefunden. Bitte installieren: brew install ffmpeg", file=sys.stderr)
        return 2

    try:
        cap = open_video(args.input)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 2

    fps = get_fps(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
    cmd = build_ffmpeg_cmd(args.output, args.input, w, h, fps, args.codec, args.bitrate, use_sw)

    proc = None
    frame_idx = 0
    try:
        nx1p, nx2p, ny1p, ny2p = [float(v) for v in args.no_pixel_zone.split(",")]
        no_pixel_enabled = True
    except Exception:
        nx1p = nx2p = ny1p = ny2p = 0.0
        no_pixel_enabled = False
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

            if no_pixel_enabled:
                nz = (
                    int(w * (nx1p / 100.0)),
                    int(h * (ny1p / 100.0)),
                    int(w * (nx2p / 100.0)),
                    int(h * (ny2p / 100.0)),
                )
            else:
                nz = None

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
                    if nz is not None and boxes_overlap((x1, y1, x2, y2), nz):
                        continue
                    pixelate_roi(frame, x1, y1, x2, y2, args.blocks)

            if proc.stdin is None:
                raise RuntimeError("ffmpeg stdin nicht verfuegbar")
            proc.stdin.write(frame.tobytes())

            frame_idx += 1
            if args.log_every > 0 and frame_idx % args.log_every == 0:
                print(f"Processed frames: {frame_idx}")

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
