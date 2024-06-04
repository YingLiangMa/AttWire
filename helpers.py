import numpy as np
import numpy as np
from skimage.draw import polygon_perimeter, line
from shapely.geometry import Polygon
from typing import Tuple, Optional


def _rotation(pts: np.ndarray, theta: float) -> np.ndarray:
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    pts = pts @ r
    return pts


def _make_box_pts(
    pos_x: float, pos_y: float, yaw: float, dim_x: float, dim_y: float
) -> np.ndarray:

    hx = dim_x / 2
    hy = dim_y / 2

    pts = np.asarray([(-hx, -hy), (-hx, hy), (hx, hy), (hx, -hy)])
    pts = _rotation(pts, yaw)
    pts += (pos_x, pos_y)
    return pts

def _get_pos(s: float) -> np.ndarray:
    return np.random.randint(10, s - 10, size=2)


def _get_yaw() -> float:
    return np.random.rand() * 2 * np.pi


def _get_size() -> int:
    return np.random.randint(18, 37)


def _get_l2w() -> float:
    return abs(np.random.normal(3 / 2, 0.2))


def _get_t2l() -> float:
    return abs(np.random.normal(1 / 3, 0.1))


def score_iou(ypred: np.ndarray, ytrue: np.ndarray) -> Optional[float]:

    assert (
        ypred.size == ytrue.size == 5
    ), "Inputs should have 5 parameters, use null array for empty predictions/labels."

    no_pred = np.any(np.isnan(ypred))
    no_label = np.any(np.isnan(ytrue))

    if no_label and no_pred:
        # true negative
        return None
    elif no_label and not no_pred:
        # false positive
        return 0
    elif not no_label and not no_pred:
        # true positive
        t = Polygon(_make_box_pts(*ytrue))
        p = Polygon(_make_box_pts(*ypred))
        iou = t.intersection(p).area / t.union(p).area
        return iou
    elif not no_label and no_pred:
        # false negative
        return 0
    else:
        raise NotImplementedError
