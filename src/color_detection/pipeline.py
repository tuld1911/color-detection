import os
import numpy as np
import cv2
from .config import DEFAULT_CONFIG
from .io_utils import read_image, save_mask, save_json, overlay_mask
from .preprocess import white_balance_grayworld, clahe_on_lab_l
from .colorspaces import to_hsv_lab
from .hsv_detect import hsv_mask
from .lab_deltae import de_mask
from .fusion import fuse_masks
from .postprocess import morph_refine, remove_small_blobs
from .palette import kmeans_palette_lab
from .stats import area_ratio_by_masks, get_all_config_colors
from .viz import COLOR_BGR_TABLE, draw_labeled_boxes, labref_to_bgr

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

        # --- Postprocess ---
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