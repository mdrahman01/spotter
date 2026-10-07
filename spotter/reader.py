"""Plate reader: fast-alpr run on the watch zone of a frame."""

from typing import NamedTuple

import numpy as np
from fast_alpr import ALPR

from spotter.config import Zone

# fast-alpr's default models, as named in its README.
DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"


class PlateRead(NamedTuple):
    text: str  # upper-case letters and digits only
    det_conf: float


def normalize(text: str) -> str:
    """Upper-case and keep only letters and digits, so "abc 1234" equals "ABC1234"."""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def zone_box(zone: Zone, width: int, height: int) -> tuple[int, int, int, int]:
    """The zone in pixels of a width x height frame: x0, y0, x1, y1."""
    x0, y0, x1, y1 = zone
    return round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)


class PlateReader:
    def __init__(self, zone: Zone, min_det_conf: float) -> None:
        self.zone = zone
        self.min_det_conf = min_det_conf
        # Both models run on the CPU execution provider. On a Mac, onnxruntime's
        # CoreML provider fails on every frame with no plate in it (the detector's
        # output then has zero elements, which CoreML cannot handle) and the
        # small models are fast enough on the CPU anyway.
        self.alpr = ALPR(
            detector_model=DETECTOR_MODEL,
            ocr_model=OCR_MODEL,
            detector_conf_thresh=min_det_conf,
            detector_providers=["CPUExecutionProvider"],
            ocr_device="cpu",
        )

    def read(self, frame: np.ndarray) -> list[PlateRead]:
        """(text, det_conf) for each plate found inside the zone.

        Plates detected below min_det_conf, or with no text read, are left out.
        """
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = zone_box(self.zone, width, height)
        crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
        reads = []
        for result in self.alpr.predict(crop):
            text = normalize(result.ocr.text) if result.ocr else ""
            if text and result.detection.confidence >= self.min_det_conf:
                reads.append(PlateRead(text, float(result.detection.confidence)))
        return reads

    def close(self) -> None:
        """Drop the ONNX sessions now, rather than leaving them to interpreter exit."""
        self.alpr = None
