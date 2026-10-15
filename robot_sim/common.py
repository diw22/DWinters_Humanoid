"""Shared helpers: JSON files and controller-to-robot coordinates."""
import json
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def board_to_robot(xy, kit):
    return (np.asarray(kit["board_origin_robot_m"]) +
            xy[0] * np.asarray(kit["board_x_robot"]) + xy[1] * np.asarray(kit["board_y_robot"]))
