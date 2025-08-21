import cv2
import numpy as np

# =========================
# Preprocess
# =========================

# Gray-world: scale kênh sao cho mean(B)=mean(G)=mean(R)
def white_balance_grayworld(bgr):
    # get 3 layers from image: blue, green, red
    b, g, r = cv2.split(bgr.astype(np.float32))

    # calculate mean of each layer
    mb, mg, mr = b.mean(), g.mean(), r.mean()

    # calculate k mean
    k = (mb + mg + mr) / 3.0

    # avoid divide by 0: if mean = 0, replace with a tiny value
    mb = max(mb, 1e-6)
    mg = max(mg, 1e-6)
    mr = max(mr, 1e-6)

    # calculate new pixel value for each layer
    b = np.clip(b * (k/mb), 0, 255)
    g = np.clip(g * (k/mg), 0, 255)
    r = np.clip(r * (k/mr), 0, 255)

    # combine 3 layers adn return (BGR, uint8)
    return cv2.merge([b, g, r]).astype(np.uint8)

# Contrast Limited Adaptive Histogram Equalization
# clip: contrast limited value, default 2.0
# tiles: local window size to process HE, default 8x8
def clahe_on_lab_l(bgr, clip=2.0, tiles=(8,8)):
    # convert BGR -> LAB: L: Lightness(độ sáng), A/B: thông tin màu
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    # split L, a, b from lab
    L, a, b = cv2.split(lab)

    # create CLAHE config
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles)

    # Apply CLAHE to channel L only, reason, just want to adjust lightness
    L2 = clahe.apply(L)

    # combine applied CLAHE L with a, b
    lab2 = cv2.merge([L2, a, b])

    # Convert LAB to BGR and return
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)