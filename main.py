#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, argparse
import numpy as np
import cv2

# =========================
# Default color config (có thể override bằng --config)
# =========================
DEFAULT_CONFIG = {
    # in HSV, H has value from 0 -> 179, red color value have value from 0 -> 10 and 170 -> 179
    # so we need to split to red1 and red2 then combine their mask to detect full range of red
    "hsv": {
        "red1":  {"h":[0,10],   "s_min":60, "v_min":40},
        "red2":  {"h":[170,179],"s_min":60, "v_min":40},
        "orange":{"h":[10,20],  "s_min":55, "v_min":45},
        "yellow":{"h":[20,35],  "s_min":55, "v_min":50},
        "green": {"h":[35,85],  "s_min":50, "v_min":35},
        "cyan":  {"h":[85,95],  "s_min":45, "v_min":35},
        "blue":  {"h":[95,140], "s_min":50, "v_min":35},
        "purple":{"h":[140,160],"s_min":45, "v_min":35},
        "pink":  {"h":[160,170],"s_min":50, "v_min":40},
        "brown": {"h":[15,25],  "s_min":50, "v_max":120}
    },
    "lab_refs": {
        # L in 0..100, a,b around -128..+127
        "red":    {"L":54,"a":80,"b":67,"dE":18},
        "orange": {"L":75,"a":23,"b":78,"dE":16},
        "yellow": {"L":97,"a":-21,"b":94,"dE":16},
        "green":  {"L":87,"a":-86,"b":83,"dE":18},
        "cyan":   {"L":91,"a":-48,"b":-14,"dE":16},
        "blue":   {"L":32,"a":79,"b":-108,"dE":20},
        "purple": {"L":45,"a":75,"b":-36,"dE":18},
        "pink":   {"L":85,"a":51,"b":-4,"dE":18},
        "brown":  {"L":37,"a":15,"b":34,"dE":18}
    },
    "fusion": {"mode":"precise","w_hsv":0.4,"w_lab":0.6}
}

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

COLOR_SET_STD = ["red","orange","yellow","green","cyan","blue","purple","pink","brown"]

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

# =========================
# Color space
# =========================
def to_hsv_lab(bgr):
    # convert BGR to LAB
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)   # H:0..179, S/V:0..255
    # convert BGR to HSV
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)   # OpenCV Lab
    return hsv, lab

# =========================
# HSV detection
# =========================

# function to determine area of input color in the image
def _in_range_hsv(hsv, h_range, s_min, v_min, v_max=255):
    # split to get channel from image
    H, S, V = cv2.split(hsv)

    # get min and max range of H config of input color
    h1, h2 = h_range

    # create binary mask by compare each value in H channel with H min and H max
    cond_h = (H >= h1) & (H <= h2)

    # create binary mask by compare each value in S, V channel with S min and between V min, V max
    cond_sv = (S.astype(np.int32) >= int(s_min)) & (V.astype(np.int32) >= int(v_min)) & (V.astype(np.int32) <= int(v_max))

    # return binary mask by combining H binary mask and SV binary mask, return unit type uint8(0-255)
    return (cond_h & cond_sv).astype(np.uint8) * 255

# function to create binary mask base on HSV value
def hsv_mask(hsv, hsv_spec, color_name):
    # determine red color: as mentioned, red color is special due to it has 2 range in HSV -> need to combine both
    if color_name == "red":
        # calculate binary mask red 1
        m1 = _in_range_hsv(hsv, hsv_spec["red1"]["h"], hsv_spec["red1"]["s_min"], hsv_spec["red1"]["v_min"])

        # calculate binary mask red 2
        m2 = _in_range_hsv(hsv, hsv_spec["red2"]["h"], hsv_spec["red2"]["s_min"], hsv_spec["red2"]["v_min"])

        # combine both by using bitwise_or and return the mask
        return cv2.bitwise_or(m1, m2)
    # other colors
    else:
        # get color spec config
        st = hsv_spec[color_name]
        v_min = st.get("v_min", 0)
        v_max = st.get("v_max", 255)

        # calculate binary mask and return
        return _in_range_hsv(hsv, st["h"], st["s_min"], v_min, v_max)

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

# =========================
# Stats
# =========================
def area_ratio_by_masks(masks_dict):
    total = 0
    areas = {}
    for k, m in masks_dict.items():
        a = int((m>0).sum())
        areas[k] = a
        total += a
    if total == 0:
        return {k: 0.0 for k in areas}
    return {k: areas[k]/total for k in areas}

def get_all_config_colors(hsv_spec, lab_refs):
    """Trả về danh sách tên màu duy nhất có trong config (hsv ∪ lab_refs).
    Tự gộp red1/red2 thành 'red'."""
    names = set(lab_refs.keys())
    for k in hsv_spec.keys():
        if k.startswith("red"):
            names.add("red")
        else:
            names.add(k)
    # loại bỏ các khóa kỹ thuật nếu có
    names.discard("red1")
    names.discard("red2")
    return sorted(names)

def labref_to_bgr(ref):
    # ref: {"L":..,"a":..,"b":..}
    L = np.uint8(np.clip(ref["L"] * 255.0/100.0, 0, 255))
    a = np.uint8(np.clip(ref["a"] + 128.0, 0, 255))
    b = np.uint8(np.clip(ref["b"] + 128.0, 0, 255))
    lab1 = np.uint8([[[L, a, b]]])
    bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)[0,0,:]
    return (int(bgr1[0]), int(bgr1[1]), int(bgr1[2]))

# =========================
# Main image pipeline
# =========================
def process_image(path, cfg, out_dir="out", k_palette=4, fusion_mode="precise", de_method="ciede2000",
                  morph_open=3, morph_close=5, min_area_ratio=0.0005, save_preview=True):
    os.makedirs(out_dir, exist_ok=True)

    hsv_spec = cfg.get("hsv", DEFAULT_CONFIG["hsv"])
    lab_refs = cfg.get("lab_refs", DEFAULT_CONFIG["lab_refs"])
    fusion_cfg = cfg.get("fusion", DEFAULT_CONFIG["fusion"])
    if fusion_mode is None:
        fusion_mode = fusion_cfg.get("mode", "precise")

    # read image
    bgr0 = read_image(path)

    # Preprocess: White Balance using gray woorld
    bgr1 = white_balance_grayworld(bgr0)

    # Preprocess: CLAHE on L channel, contrast limit at 2.0 and using tiles 8x8
    bgr2 = clahe_on_lab_l(bgr1, clip=2.0, tiles=(8,8))

    # get hsv and lab from preprocessed image
    hsv, lab = to_hsv_lab(bgr2)

    # get unique color names form lab_refs and hsv_spec (note: remove red1 and red2, only red)
    color_names = get_all_config_colors(hsv_spec, lab_refs)

    # get draw color which will use to annotate
    draw_colors = COLOR_BGR_TABLE.copy()
    for cname in color_names:
        if cname not in draw_colors and cname in lab_refs:
            draw_colors[cname] = labref_to_bgr(lab_refs[cname])

    # create empty masks
    masks = {}

    # get Height and Width from input image, use for masking
    H, W = hsv.shape[:2]

    for name in color_names:
        # --- HSV branch ---
        # red is special case: have to merge two range of red in HSV
        if name == "red":
            if "red1" in hsv_spec and "red2" in hsv_spec:
                m_hsv = hsv_mask(hsv, hsv_spec, "red")  # dùng union red1, red2
            else:
                # nếu config không có red1/red2 -> cho nhánh HSV "trắng" để AND/OR không triệt
                m_hsv = np.full((H, W), 255, np.uint8)
        else:
            if name in hsv_spec:
                m_hsv = hsv_mask(hsv, hsv_spec, name)
            else:
                # thiếu HSV -> để nhánh HSV là "trắng", nhánh Lab quyết định
                m_hsv = np.full((H, W), 255, np.uint8)

        # --- Lab ΔE branch ---
        lab_name = name if name in lab_refs else ("red" if "red" in lab_refs else name)
        m_lab = de_mask(lab, lab_refs, lab_name, method=de_method)

        # --- Fuse (precise = AND, recall = OR) ---
        m = fuse_masks(m_hsv, m_lab, mode=fusion_mode,
                       w_hsv=fusion_cfg.get("w_hsv", 0.4),
                       w_lab=fusion_cfg.get("w_lab", 0.6))

        # --- Hậu xử lý ---
        m = morph_refine(m, k_open=morph_open, k_close=morph_close)
        m = remove_small_blobs(m, min_area_ratio=min_area_ratio)

        masks[name] = m
        save_mask(os.path.join(out_dir, f"mask_{name}.png"), m)

    # Palette
    pal = kmeans_palette_lab(bgr2, k=k_palette, resize_long=480)
    save_json(os.path.join(out_dir, "palette.json"), pal)

    # Ratios
    ratios = area_ratio_by_masks(masks)
    save_json(os.path.join(out_dir, "ratios.json"), ratios)

    # Preview overlay một số màu tiêu biểu
    if save_preview and "red" in masks:
        ov = overlay_mask(bgr2, masks["red"], alpha=0.5, color_bgr=(0,0,255))
        cv2.imwrite(os.path.join(out_dir, "preview_red.png"), ov)
    if save_preview and "green" in masks:
        ov2 = overlay_mask(bgr2, masks["green"], alpha=0.5, color_bgr=(0,255,0))
        cv2.imwrite(os.path.join(out_dir, "preview_green.png"), ov2)
        # === Annotate: vẽ bbox + label màu ===
        # Chọn Top-3 màu theo area ratio (bỏ các màu ratio == 0)
        sorted_colors = sorted(ratios.items(), key=lambda kv: kv[1], reverse=True)
        top3 = [name for name, r in sorted_colors[:5] if r > 0]

        # Nếu ảnh có <3 màu, vẫn hoạt động bình thường
        masks_top = {name: masks[name] for name in top3}

        annotated = draw_labeled_boxes(
            bgr2,
            masks_top,
            min_area_ratio=0.001,
            thickness=2
        )
        cv2.imwrite(os.path.join(out_dir, "annotated.png"), annotated)

    # === In kết quả ra console (đẹp, dễ đọc) ===
    print("\n== KẾT QUẢ NHẬN DIỆN MÀU ==")
    print("• Tỷ lệ theo màu (trên toàn khung):")
    for k in sorted(ratios.keys()):
        print(f"  - {k:<7}: {ratios[k]*100:5.1f}%")
    print("• Palette chủ đạo (k-center, HEX ~ tỉ lệ):")
    for p in pal:
        print(f"  - {p['hex']}  ~ {p['ratio']*100:4.1f}%")

    # (giữ nguyên phần return)
    return {"palette": pal, "ratios": ratios}

def load_config(path):
    if path is None:
        return DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # hợp nhất nhẹ để tránh thiếu khóa
    cfg_out = DEFAULT_CONFIG.copy()
    cfg_out.update(cfg)
    return cfg_out

def main():
    # --- cấu hình nhanh khi chạy trong IDE ---
    image_path = "shapes.png"
    out_dir = "out"
    cfg_path = None           # hoặc "configs/colors.json"
    k_palette = 4
    fusion_mode = "recall"   # "precise" hoặc "recall"
    de_method = "ciede2000"   # "ciede2000" hoặc "de76"

    cfg = load_config(cfg_path)
    result = process_image(
        image_path, cfg,
        out_dir=out_dir,
        k_palette=k_palette,
        fusion_mode=fusion_mode,
        de_method=de_method,
        morph_open=3,
        morph_close=5,
        min_area_ratio=0.0005,  # lọc blob nhỏ cho mask (trước khi vẽ bbox)
        save_preview=True
    )
    # Kết quả cũng đã được in trong process_image()

if __name__ == "__main__":
    main()
