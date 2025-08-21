import cv2
import numpy as np

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