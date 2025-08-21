import cv2
import numpy as np

def morph_refine(mask, k_open=3, k_close=5):
    k1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open,k_open))
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close,k_close))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k2)
    return m

def remove_small_blobs(mask, min_area_ratio=0.0005):
    h, w = mask.shape[:2]
    min_area = int(min_area_ratio * h * w)
    if min_area <= 0:
        return mask
    bw = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):  # 0 là nền
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels==i] = 255
    return out