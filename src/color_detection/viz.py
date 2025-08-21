import cv2
import numpy as np

# ==== Color table cho vẽ khung/label (BGR) ====
COLOR_BGR_TABLE = {
    "red":    (0, 0, 255),
    "orange": (0, 128, 255),
    "yellow": (0, 225, 255),
    "green":  (0, 200, 0),
    "cyan":   (255, 255, 0),
    "blue":   (255, 0, 0),
    "purple": (200, 0, 200),
    "pink":   (203, 120, 255),
    "brown":  (19, 69, 139),
    "gray":   (128, 128, 128),
    "white":  (235, 235, 235),
    "black":  (10, 10, 10)
}

def _draw_text_with_bg(img, text, org, fg=(255,255,255), bg=(0,0,0), scale=0.6, thickness=1, pad=3):
    """Vẽ chữ có nền mờ để dễ đọc trên mọi nền."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x, y - th - 2*pad), (x + tw + 2*pad, y + base + pad), bg, thickness=-1)
    cv2.putText(img, text, (x + pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thickness, cv2.LINE_AA)

def draw_labeled_boxes(bgr, masks_dict, min_area_ratio=0.001, thickness=2, color_table=None):
    """
    Vẽ bbox cho từng 'màu' dựa trên mask.
    - Lọc blob nhỏ < min_area_ratio * (H*W).
    - Label: <color> <pct_of_frame>%.
    """
    if color_table is None:
        color_table = COLOR_BGR_TABLE
    out = bgr.copy()
    H, W = out.shape[:2]
    min_area = int(min_area_ratio * H * W)

    # Tính % diện tích theo khung hình để in trên label
    pct_by_color = {}
    for name, m in masks_dict.items():
        pct_by_color[name] = float((m > 0).sum()) / float(H * W) if H*W > 0 else 0.0

    for name, mask in masks_dict.items():
        if mask is None:
            continue
        bw = (mask > 0).astype(np.uint8)
        if bw.sum() == 0:
            continue

        num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
        color = color_table.get(name, (255,255,255))
        for i in range(1, num):  # 0 là nền
            x, y, w, h, area = stats[i]
            if area < min_area:
                continue
            cv2.rectangle(out, (x, y), (x+w, y+h), color, thickness)
            # Label
            pct_txt = f"{pct_by_color.get(name,0.0)*100:.1f}%"
            text = f"{name} {pct_txt}"
            # nền label lấy màu đậm hơn một chút để tương phản
            bg = tuple(int(c*0.6) for c in color)
            _draw_text_with_bg(out, text, (x, max(20, y-6)), fg=(255,255,255), bg=bg, scale=0.6, thickness=1)

    return out

def labref_to_bgr(ref):
    # ref: {"L":..,"a":..,"b":..}
    L = np.uint8(np.clip(ref["L"] * 255.0/100.0, 0, 255))
    a = np.uint8(np.clip(ref["a"] + 128.0, 0, 255))
    b = np.uint8(np.clip(ref["b"] + 128.0, 0, 255))
    lab1 = np.uint8([[[L, a, b]]])
    bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)[0,0,:]
    return (int(bgr1[0]), int(bgr1[1]), int(bgr1[2]))