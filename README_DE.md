# Plater – Kennzeichen in 4K-Videos verpixeln (Apple Silicon)

Dieses Projekt erkennt Kfz-Kennzeichen in 4K-Videos und verpixelt sie automatisch. Optimiert fuer Apple Silicon (M4) mit MPS, Fallback auf CPU. Fokus auf Datenschutz: lieber zu viel verpixeln als Kennzeichen zu uebersehen.

Englische Anleitung: siehe `README_EN.md`.

## Voraussetzungen
- Python 3.10+
- ffmpeg via Homebrew: `brew install ffmpeg`
- Ein YOLOv8-Kennzeichenmodell als `.pt` Datei (z. B. `best.pt`)

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Schnellstart (einfach)
1) Terminal im Projektordner oeffnen.
2) Virtuelle Umgebung aktivieren:
```bash
source .venv/bin/activate
```
3) Ausfuehren (Dateinamen anpassen):
```bash
python blur_plates_m4.py --input input.mp4 --output output.mp4 --weights best.pt
```

Wenn dir beim Start Fehlermeldungen wie `ModuleNotFoundError: No module named 'cv2'` erscheinen, fehlt die Umgebung. Dann nutze die Setup-Schritte oben.

Wenn du ohne Parameter startest, zeigt das Programm eine kurze, leicht verstaendliche Hilfe an.

## Woher bekomme ich `best.pt`?
- Trainiere ein eigenes YOLOv8-Kennzeichenmodell und exportiere es als `.pt`.
- Nutze ein bestehendes Kennzeichen-Detektionsmodell von einem vertrauenswuerdigen Anbieter (achte auf Lizenz und Datenschutz).
- Beispiel: Ein passendes Modell gibt es z. B. hier: https://huggingface.co/Koushim/yolov8-license-plate-detection/tree/main
- Wichtig: Das Modell muss Kennzeichen als Objekte erkennen (keine OCR noetig).

## Beispiele
HEVC Default (4K, MPS, Audio wird uebernommen):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output.mp4 \
  --weights /path/to/plate_model.pt
```

H264 kompatibler Output (laeuft fast ueberall, empfehle 50M fuer beste Qualitaet):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output_h264.mp4 \
  --weights /path/to/plate_model.pt \
  --codec h264 \
  --bitrate 50M
```

Software-Encoding erzwingen (wenn Hardware-Encoding zickt):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output_sw.mp4 \
  --weights /path/to/plate_model.pt \
  --force_sw
```

Nur Schnelltest mit kleinerer Arbeitsaufloesung (schneller, weniger genau):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output_fast.mp4 \
  --weights /path/to/plate_model.pt \
  --work_w 1280
```

Preset fuer hohe Qualitaet (langsamer, bessere Erkennung):
```bash
python blur_plates_m4.py \
  --input input.mp4 \
  --output output_quality.mp4 \
  --weights /path/to/plate_model.pt \
  --preset quality
```

## Wichtige Parameter
- `work_w`: Arbeitsbreite fuer Detektion (z.B. 1280 oder 1920). 0 = Originalaufloesung.
- `imgsz`: YOLO Inferenzgroesse (groesser = bessere Erkennung, aber langsamer).
- `conf`: Confidence Threshold (niedriger = mehr Treffer).
- `blocks`: Pixelblock-Groesse (kleiner = grober, staerkerer Effekt).
- `pad`: Sicherheitsrand in Pixeln um jede Box.
- `no_pixel_zone`: No-Pixel-Zone in Prozent als `x1,x2,y1,y2` (Default `0,22,59,100` fuer HUD unten links).
- `no_pixel_zone2`: Zweite No-Pixel-Zone (Default `78,100,59,100` fuer HUD unten rechts).
- `force_sw`: Software-Encoding erzwingen (nuetzlich, wenn VideoToolbox zickt).
- `test_minutes`: Nur die ersten N Minuten verarbeiten (0 = alles).
- `preset`: `fast`, `balanced`, `quality` fuer einfache Speed/Qualitaets-Wahl.
- `debug_overlay`: Zeichnet Boxen zur Kontrolle ins Video.
- `bitrate`: Standard ist `auto` (uebernimmt Bitrate vom Input), alternativ z. B. `50M`.

## Bedienung in einfachen Worten
1) Lege dein Video (z. B. `input.mp4`) und die Gewichte (z. B. `best.pt`) in den Projektordner.
2) Oeffne ein Terminal im Projektordner.
3) Starte das Programm wie im Schnellstart gezeigt.
4) Danach findest du die Ausgabe als `output.mp4` im selben Ordner.
Tipp: Wenn `best.pt` im Ordner liegt und du `--weights` vergisst, wird es automatisch genutzt.

## FAQ
Wie sicher ist die Verpixelung?
Das Ziel ist Unlesbarkeit. Nutze kleine `blocks` und ausreichend `pad`, um nichts zu uebersehen.

Warum wird mein HUD/eine Anzeige verpixelt?
Nutze `--no_pixel_zone`, um feste Bereiche auszunehmen (z. B. unten links).

Warum ist das Ergebnis langsam?
4K + YOLO ist rechenintensiv. Setze `work_w` kleiner (z. B. 1280) und pruefe die Geschwindigkeit.

Warum funktioniert Hardware-Encoding nicht?
Wenn VideoToolbox zickt, nutze `--force_sw`. Siehe Troubleshooting.

Gibt es Audio im Output?
Ja, Audio wird vom Input uebernommen (wenn vorhanden).

## Hinweis zu Insta360
Empfehlung: In Insta360 Studio zuerst reframen/flat exportieren (16:9), dann mit diesem Tool verpixeln.

## Troubleshooting
VideoToolbox Fehler -12908 (HW-Encoding schlaegt fehl): Ursache ist oft Pixel-Format-Negotiation. Stelle sicher, dass ein VT-kompatibles Format (nv12) genutzt wird. Das Script setzt dies automatisch fuer VideoToolbox. Falls es dennoch scheitert, teste per ffmpeg:
```bash
ffmpeg -y -f lavfi -i testsrc2=size=3840x2160:rate=60 -t 2 -vf format=nv12 -c:v hevc_videotoolbox -b:v 12M vt_test.mp4
```

Rotation wirkt falsch: Manche Dateien haben Rotations-Metadaten. Normalisieren via ffmpeg:
```bash
ffmpeg -i input.mp4 -vf "transpose=0" -c:a copy normalized.mp4
```

Variable Framerate: Empfohlen ist eine Normalisierung vorab:
```bash
ffmpeg -i input.mp4 -vsync cfr -r 25 -c:v libx264 -c:a copy normalized.mp4
```

## Datenschutz-Hinweis
Ziel ist Unlesbarkeit. Nutze grobe Pixel (`blocks` klein) und ausreichend `pad`, damit nichts uebersehen wird.
