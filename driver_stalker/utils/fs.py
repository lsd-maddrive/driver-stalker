"""Contain functions for interacting with file system."""

from collections.abc import Sequence
import itertools
from pathlib import Path

import cv2
from natsort import natsorted
import numpy as np
import numpy.typing as npt


def read_image(fpath: str | Path, gray_scale: bool = False) -> npt.NDArray[np.uint8]:
    """Read image from a file system."""
    if isinstance(fpath, Path):
        fpath = str(fpath)

    if gray_scale:
        return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE).astype(np.uint8)

    img = cv2.imread(fpath, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)


def get_image_fpaths_from_folder(dpath: str | Path) -> Sequence[Path]:
    """Get paths of images from the specified folder.

    Notes
    -----
    Recursive

    """
    if isinstance(dpath, str):
        dpath = Path(dpath)

    supported_extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.png"]
    return natsorted(
        itertools.chain.from_iterable(dpath.rglob(pattern) for pattern in supported_extensions),
    )
