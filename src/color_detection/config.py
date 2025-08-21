import json
from pathlib import Path

DEFAULT_CONFIG = {
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

def load_config(path):
    if path is None:
        return DEFAULT_CONFIG
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    out = DEFAULT_CONFIG.copy()
    out.update(cfg)
    return out