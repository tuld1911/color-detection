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