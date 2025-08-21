import numpy as np

# =========================
# ΔE helpers (Lab)
# =========================
def lab_split_float(lab_img):
    # OpenCV Lab: L in 0..255 (maps ~0..100), a,b in 0..255 with 128 offset
    L = lab_img[:,:,0].astype(np.float32) * (100.0/255.0)
    a = lab_img[:,:,1].astype(np.float32) - 128.0
    b = lab_img[:,:,2].astype(np.float32) - 128.0
    return L, a, b

def deltaE76(lab_img, Lr, ar, br):
    L, a, b = lab_split_float(lab_img)
    dL = L - float(Lr)
    da = a - float(ar)
    db = b - float(br)
    return np.sqrt(dL*dL + da*da + db*db)

# CIEDE2000
# calculate deltaE: which indicate the difference between original color and color in the input image
def deltaE2000(lab_img, Lr, ar, br):
    # Split to 3 channels
    L, a, b = lab_split_float(lab_img)

    # Convert reference to arrays for broadcast
    Lr = np.array(Lr, dtype=np.float32)
    ar = np.array(ar, dtype=np.float32)
    br = np.array(br, dtype=np.float32)

    # Weighting factors
    kL = 1.0; kC = 1.0; kH = 1.0

    # Compute C'
    C1 = np.sqrt(a*a + b*b)
    C2 = np.sqrt(ar*ar + br*br)
    Cbar = (C1 + C2) * 0.5

    G = 0.5 * (1 - np.sqrt((Cbar**7) / (Cbar**7 + 25**7)))
    a1p = (1 + G) * a
    a2p = (1 + G) * ar
    C1p = np.sqrt(a1p*a1p + b*b)
    C2p = np.sqrt(a2p*a2p + br*br)

    # angles
    hp_f = lambda x, y: np.degrees(np.arctan2(y, x)) % 360.0
    h1p = hp_f(a1p, b)
    h2p = hp_f(a2p, br)

    dLp = L - Lr
    dCp = C1p - C2p

    # dhp
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, dhp)
    dhp = np.where(dhp < -180, dhp + 360, dhp)

    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    # averages
    Lbarp = (L + Lr) * 0.5
    Cbarp = (C1p + C2p) * 0.5

    # hbarp
    hsum = h1p + h2p
    habsp = np.abs(h1p - h2p)
    hbarp = np.where((habsp <= 180), hsum * 0.5, np.where(hsum < 360, (hsum + 360) * 0.5, (hsum - 360) * 0.5))

    T = (1
         - 0.17*np.cos(np.radians(hbarp - 30))
         + 0.24*np.cos(np.radians(2*hbarp))
         + 0.32*np.cos(np.radians(3*hbarp + 6))
         - 0.20*np.cos(np.radians(4*hbarp - 63)))

    Sl = 1 + (0.015 * (Lbarp - 50)**2) / np.sqrt(20 + (Lbarp - 50)**2)
    Sc = 1 + 0.045 * Cbarp
    Sh = 1 + 0.015 * Cbarp * T

    delthetarad = np.radians(30) * np.exp(-(( (hbarp - 275)/25 )**2))
    Rc = 2 * np.sqrt((Cbarp**7) / (Cbarp**7 + 25**7))
    Rt = -np.sin(2 * delthetarad) * Rc

    dE = np.sqrt(
        (dLp / (kL*Sl))**2 +
        (dCp / (kC*Sc))**2 +
        (dHp / (kH*Sh))**2 +
        Rt * (dCp / (kC*Sc)) * (dHp / (kH*Sh))
    )
    return dE

# function to create binary mask base on deltaE
def de_mask(lab_img, lab_refs, name, method="ciede2000"):
    ref = lab_refs[name if name in lab_refs else "red"]
    if method == "ciede2000":
        dE = deltaE2000(lab_img, ref["L"], ref["a"], ref["b"])
    else:
        dE = deltaE76(lab_img, ref["L"], ref["a"], ref["b"])
    thr = float(ref.get("dE", 18))
    return (dE < thr).astype(np.uint8) * 255
