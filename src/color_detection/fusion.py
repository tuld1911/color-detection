import cv2

# =========================
# Fusion & Postprocess
# =========================
def fuse_masks(m_hsv, m_lab, mode="precise", w_hsv=0.4, w_lab=0.6):
    if mode == "precise":
        return cv2.bitwise_and(m_hsv, m_lab)
    elif mode == "recall":
        return cv2.bitwise_or(m_hsv, m_lab)
    else:
        # weighted soft có thể thêm sau; hiện ưu tiên precise/recall
        return cv2.bitwise_and(m_hsv, m_lab)