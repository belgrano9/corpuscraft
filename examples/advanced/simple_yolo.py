"""
Minimal YOLO detection test.

Accepts a PDF (renders page 1 with pypdfium2) or any image file (loads directly).
Uses the DocLayNet checkpoint so Table/Picture/Figure labels are active.

Usage:
    uv run examples/advanced/simple_yolo.py
"""

from pathlib import Path

from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download, list_repo_files
from PIL import Image

SOURCE_FILE = Path(r"E:\workspace\CorpusCraft\scratch\bulletin-de-paie-du-011025-au-311025-page-1.png")
OUTPUT_FILE = Path("scratch") / "yolo_result.jpg"
MODEL_ID = "juliozhao/DocLayout-YOLO-DocLayNet-Docsynth300K_pretrained"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Load image — PDF renders page 1, anything else opens directly
if SOURCE_FILE.suffix.lower() == ".pdf":
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(SOURCE_FILE))
    pil_image = pdf[0].render(scale=150 / 72.0).to_pil().convert("RGB")
else:
    pil_image = Image.open(SOURCE_FILE).convert("RGB")

print(f"Input: {SOURCE_FILE.name}  size={pil_image.size}")

# Run YOLO
pt = next(f for f in list_repo_files(MODEL_ID) if f.endswith(".pt"))
model = YOLOv10(str(hf_hub_download(repo_id=MODEL_ID, filename=pt)))
det_res = model.predict(pil_image, imgsz=1024, conf=0.2, device="cpu")

# plot() returns BGR numpy array
annotated_bgr = det_res[0].plot(line_width=3, font_size=14)
Image.fromarray(annotated_bgr[..., ::-1]).save(str(OUTPUT_FILE))

print(f"Detections: {len(det_res[0].boxes)}")
print(f"Saved: {OUTPUT_FILE.absolute()}")
