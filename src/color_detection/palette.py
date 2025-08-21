import cv2
import numpy as np

# =========================
# Palette via cv2.kmeans (không cần sklearn)
# =========================
def kmeans_palette_lab(bgr, k=4, resize_long=480, attempts=5):
    h, w = bgr.shape[:2]
    scale = resize_long / max(h, w)
    if scale < 1.0:
        bgr_small = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        bgr_small = bgr

    lab = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2LAB)
    X = lab.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.1)
    compactness, labels, centers = cv2.kmeans(
        X, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS
    )

    # ratio mỗi cụm
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    ratios = (counts / counts.sum()).tolist()

    # chuyển center Lab -> HEX
    hexs = []
    for c in centers:
        L,a,b = c
        lab1 = np.uint8([[[L, a, b]]])
        bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)[0,0,:]
        rgb = bgr1[::-1]  # BGR->RGB
        hexs.append('#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2])))

    # sắp xếp theo tỉ lệ giảm dần
    order = np.argsort(ratios)[::-1]
    return [{"hex":hexs[i], "ratio":float(ratios[i])} for i in order]