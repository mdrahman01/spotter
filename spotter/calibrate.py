"""Find where a plate appears in a new camera view and propose a watch zone.

    python -m spotter.calibrate --source rtsp://CAMERA/stream --expect PLATE [--seconds 30]
    python -m spotter.calibrate --source samples/IMG_2472.mov --expect PLATE

For --seconds (default 30) it scans whole frames in overlapping tiles, so small
plates are found too, and keeps every read of the expected plate. It prints the
plate's width in pixels (lowest, middle, highest) and a proposed zone that covers
every place the plate was read, with a margin; it saves a preview under data/
with the current zone, the proposed zone and the plate boxes drawn on it; and it
prints the SPOTTER_ZONE line for .env. It changes no setting itself, and it says
so plainly when the plate was not read anywhere. No plate text is printed.
"""

import argparse
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from fast_alpr import ALPR

from spotter import config
from spotter.config import Settings, Zone
from spotter.live import FrameSource, is_stream, mask_source
from spotter.reader import DETECTOR_MODEL, OCR_MODEL, normalize, zone_box

REPO_ROOT = Path(__file__).resolve().parent.parent
PREVIEW_PATH = REPO_ROOT / "data" / "calibrate_preview.jpg"
TILE = 640  # pixels; about the size at which the detector sees a plate well
OVERLAP = 0.5
SMALL_PLATE_PX = 60  # below this width the reader has little margin
SAMPLE_FPS = 2.0


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def tiles(width: int, height: int, size: int = TILE, overlap: float = OVERLAP) -> list[tuple[int, int, int, int]]:
    """Square tiles covering the frame with the given overlap, plus the whole frame as one tile.

    A frame no bigger than a tile is scanned whole.
    """
    if size >= width and size >= height:
        return [(0, 0, width, height)]
    size = min(size, width, height)
    stride = max(1, int(size * (1 - overlap)))
    xs = list(range(0, max(1, width - size), stride)) + [width - size]
    ys = list(range(0, max(1, height - size), stride)) + [height - size]
    boxes = {(x, y, x + size, y + size) for x in xs for y in ys}
    boxes.add((0, 0, width, height))
    return sorted(boxes)


def iou(a: Box, b: Box) -> float:
    inter_w = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    inter_h = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = inter_w * inter_h
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union else 0.0


def dedupe(boxes: list[Box]) -> list[Box]:
    """One box per plate: overlapping reads from neighbouring tiles keep the most confident."""
    kept: list[Box] = []
    for box in sorted(boxes, key=lambda b: -b.confidence):
        if all(iou(box, other) < 0.5 for other in kept):
            kept.append(box)
    return kept


def scan_frame(alpr: ALPR, frame: np.ndarray, expected: str) -> list[Box]:
    """Every place the expected plate is read in the frame, in frame pixels."""
    height, width = frame.shape[:2]
    found = []
    for x0, y0, x1, y1 in tiles(width, height):
        for result in alpr.predict(np.ascontiguousarray(frame[y0:y1, x0:x1])):
            if result.ocr and normalize(result.ocr.text) == expected:
                b = result.detection.bounding_box
                found.append(Box(x0 + b.x1, y0 + b.y1, x0 + b.x2, y0 + b.y2, float(result.detection.confidence)))
    return dedupe(found)


def propose_zone(boxes: list[Box], width: int, height: int) -> Zone:
    """A zone covering every box, with a margin of one plate width sideways and two plate heights up and down."""
    plate_w = statistics.median(b.width for b in boxes)
    plate_h = statistics.median(b.height for b in boxes)
    x0 = max(0, min(b.x1 for b in boxes) - plate_w)
    x1 = min(width, max(b.x2 for b in boxes) + plate_w)
    y0 = max(0, min(b.y1 for b in boxes) - 2 * plate_h)
    y1 = min(height, max(b.y2 for b in boxes) + 2 * plate_h)
    return (
        max(0.0, round(x0 / width - 0.005, 2)),
        max(0.0, round(y0 / height - 0.005, 2)),
        min(1.0, round(x1 / width + 0.005, 2)),
        min(1.0, round(y1 / height + 0.005, 2)),
    )


def width_stats(boxes: list[Box]) -> tuple[int, int, int]:
    widths = sorted(b.width for b in boxes)
    return widths[0], int(statistics.median(widths)), widths[-1]


def draw_preview(frame: np.ndarray, current: Zone, proposed: Zone | None, boxes: list[Box]) -> np.ndarray:
    preview = frame.copy()
    height, width = preview.shape[:2]
    thickness = max(2, round(width / 400))
    x0, y0, x1, y1 = zone_box(current, width, height)
    cv2.rectangle(preview, (x0, y0), (x1, y1), (255, 128, 0), thickness)  # current zone: blue
    cv2.putText(preview, "current zone", (x0 + 5, max(y0 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 128, 0), 2)
    for b in boxes:
        cv2.rectangle(preview, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 255), max(1, thickness // 2))  # plate: yellow
    if proposed:
        x0, y0, x1, y1 = zone_box(proposed, width, height)
        cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 220, 0), thickness)  # proposed zone: green
        cv2.putText(preview, "proposed zone", (x0 + 5, min(y1 + 25, height - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)
    return preview


def frames(source: str, seconds: float, live: bool):
    """Sampled frames from the first `seconds` of a file, or `seconds` of a live source."""
    if live:
        reader = FrameSource(source)
        reader.start()
        deadline = time.time() + seconds
        last_seq = None
        try:
            while time.time() < deadline and not reader.ended:
                newest = reader.newest()
                if newest is not None and newest[2] != last_seq:
                    last_seq = newest[2]
                    yield newest[0]
                time.sleep(1 / SAMPLE_FPS)
        finally:
            reader.stop()
        return
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / SAMPLE_FPS))
    index = 0
    try:
        while capture.grab() and index / fps < seconds:
            if index % step == 0:
                ok, frame = capture.retrieve()
                if ok:
                    yield frame
            index += 1
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m spotter.calibrate", description="Propose a watch zone for a camera view.")
    parser.add_argument("--source", help="video file or rtsp:// URL (default: SPOTTER_SOURCE)")
    parser.add_argument("--expect", required=True, metavar="PLATE", help="your plate text, from the command line only")
    parser.add_argument("--seconds", type=float, default=30, help="how long to look (default 30)")
    parser.add_argument("--live", action="store_true", help="treat a file as a camera (rtsp:// is always live)")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")
    settings: Settings = config.load({"source": args.source})
    if not settings.source:
        parser.error("no source: pass --source or set SPOTTER_SOURCE")
    expected = normalize(args.expect)
    if not expected:
        parser.error("--expect needs at least one letter or digit")
    live = args.live or is_stream(settings.source)

    print(f"source: {mask_source(settings.source)} | looking for {args.seconds:g}s | tiles of {TILE} px with {OVERLAP:.0%} overlap")
    alpr = ALPR(
        detector_model=DETECTOR_MODEL,
        ocr_model=OCR_MODEL,
        detector_conf_thresh=settings.min_det_conf,
        detector_providers=["CPUExecutionProvider"],
        ocr_device="cpu",
    )
    boxes: list[Box] = []
    scanned = 0
    last_frame = best_frame = None
    try:
        for frame in frames(settings.source, args.seconds, live):
            scanned += 1
            last_frame = frame
            found = scan_frame(alpr, frame, expected)
            if found:
                boxes += found
                best_frame = frame
    except RuntimeError as err:
        print(err, file=sys.stderr)
        return 2
    if last_frame is None:
        print("no frames came from the source; nothing to calibrate")
        return 2
    height, width = last_frame.shape[:2]
    current = settings.zone
    print(f"frames scanned: {scanned} ({width}x{height}) | reads of the plate: {len(boxes)}")
    print(f"current zone: {','.join(str(v) for v in current)} = x {zone_box(current, width, height)[0]}-{zone_box(current, width, height)[2]}, y {zone_box(current, width, height)[1]}-{zone_box(current, width, height)[3]} px")

    proposed = None
    if boxes:
        low, middle, high = width_stats(boxes)
        print(f"plate width in pixels: lowest {low}, middle {middle}, highest {high}")
        if middle < SMALL_PLATE_PX:
            print(f"note: the plate is small here ({middle} px wide); the reader worked, but with little margin")
        proposed = propose_zone(boxes, width, height)
        px = zone_box(proposed, width, height)
        print(f"proposed zone: {','.join(str(v) for v in proposed)} = x {px[0]}-{px[2]}, y {px[1]}-{px[3]} px (covers every read, with a margin)")
        print(f"for .env:  SPOTTER_ZONE={','.join(str(v) for v in proposed)}")
    else:
        print("the plate was not read anywhere: it is too small at this distance, or out of view. No zone proposed.")
    PREVIEW_PATH.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(PREVIEW_PATH), draw_preview(best_frame if best_frame is not None else last_frame, current, proposed, boxes))
    print(f"preview saved: {PREVIEW_PATH.relative_to(REPO_ROOT)} (current zone blue, proposed zone green, plate boxes yellow)")
    print("no setting was changed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
