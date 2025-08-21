import cv2

# =========================
# Color space
# =========================
def to_hsv_lab(bgr):
    # convert BGR to LAB
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)   # H:0..179, S/V:0..255
    # convert BGR to HSV
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)   # OpenCV Lab
    return hsv, lab