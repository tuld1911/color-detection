import os, json
import cv2
import numpy as np

# =========================
# I/O helpers
# =========================
def read_image(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imread failed")
    return img  # BGR uint8

def save_mask(path, mask_uint8):
    cv2.imwrite(path, mask_uint8)

def overlay_mask(img_bgr, mask, alpha=0.5, color_bgr=(0,0,255)):
    ov = img_bgr.copy()
    sel = mask > 0
    if sel.any():
        ov[sel] = (ov[sel] * (1-alpha) + np.array(color_bgr, dtype=np.float32) * alpha).astype(np.uint8)
    return ov

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)