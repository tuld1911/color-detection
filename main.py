from src.color_detection import load_config, process_image

def main():
    image_path = "shapes.png"
    out_dir = "out"
    cfg_path = None
    k_palette = 4
    fusion_mode = "recall"
    de_method = "ciede2000"

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
