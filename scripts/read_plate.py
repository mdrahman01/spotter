"""Probe: can fast-alpr read the licence plate in the driveway sample photos?

Every image in samples/ is read twice: at full size, and as a copy downscaled
to 2560 px wide to mimic the driveway camera's resolution. Each run prints one
line: file name, variant (full or 2560), plate text read, detection confidence,
OCR confidence and milliseconds taken, or NO PLATE FOUND. An annotated copy
with the detection box is saved to samples_out/.

    .venv/bin/python scripts/read_plate.py --expect <true plate text>

With --expect, each line ends in MATCH or MISS and a final count is printed.
The photos show a real plate: samples/ and samples_out/ are gitignored, and the
true plate text is only ever passed on the command line.
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
from fast_alpr import ALPR

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = REPO_ROOT / "samples"
OUT_DIR = REPO_ROOT / "samples_out"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
CAMERA_WIDTH = 2560

# fast-alpr's default models, as named in its README.
DETECTOR_MODEL = "yolo-v9-t-384-license-plate-end2end"
OCR_MODEL = "cct-xs-v2-global-model"


def normalize(plate: str) -> str:
    """Upper-case and keep only letters and digits, so "abc 1234" equals "ABC1234"."""
    return "".join(ch for ch in plate.upper() if ch.isalnum())


def variants(image):
    """Yield (name, frame) for the full-size image and its 2560 px wide copy.

    An image that is already 2560 px wide or narrower is not upscaled.
    """
    yield "full", image
    height, width = image.shape[:2]
    if width > CAMERA_WIDTH:
        size = (CAMERA_WIDTH, round(height * CAMERA_WIDTH / width))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    yield str(CAMERA_WIDTH), image


def pick(results, expected):
    """The plate to report: the one matching --expect if any, else the most
    confident detection."""
    for result in results:
        if expected and result.ocr and normalize(result.ocr.text) == expected:
            return result
    return max(results, key=lambda result: result.detection.confidence)


def ocr_confidence(ocr) -> float | None:
    """Mean of the per-character confidences, which is how fast-alpr reports it."""
    confidence = ocr.confidence
    if isinstance(confidence, list):
        return statistics.mean(confidence) if confidence else None
    return confidence


def save_annotated(frame, results, path: Path) -> None:
    """Save a copy of the frame with a box around every detected plate."""
    annotated = frame.copy()
    thickness = max(2, round(annotated.shape[1] / 400))
    for result in results:
        box = result.detection.bounding_box
        cv2.rectangle(
            annotated, (box.x1, box.y1), (box.x2, box.y2), (36, 255, 12), thickness
        )
    if not cv2.imwrite(str(path), annotated):
        print(f"could not write {path}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read the plate in every samples/ photo with fast-alpr."
    )
    parser.add_argument(
        "--expect",
        metavar="PLATE",
        help="true plate text; adds MATCH or MISS to each line and a final count",
    )
    args = parser.parse_args()
    expected = normalize(args.expect) if args.expect is not None else None
    if expected == "":
        parser.error("--expect needs at least one letter or digit")

    paths = []
    if SAMPLES_DIR.is_dir():
        paths = sorted(
            p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
    if not paths:
        print(f"no .jpg, .jpeg or .png images in {SAMPLES_DIR}", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(exist_ok=True)

    alpr = ALPR(detector_model=DETECTOR_MODEL, ocr_model=OCR_MODEL)
    print(f"detector: {DETECTOR_MODEL} | ocr: {OCR_MODEL} | {len(paths)} images")

    runs = matches = 0
    warmed_up = False
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"{path.name} | UNREADABLE IMAGE")
            continue
        if not warmed_up:
            # Untimed first call, so model start-up does not inflate the first
            # line's milliseconds.
            alpr.predict(image)
            warmed_up = True

        for variant, frame in variants(image):
            start = time.perf_counter()
            results = alpr.predict(frame)
            elapsed_ms = (time.perf_counter() - start) * 1000
            save_annotated(frame, results, OUT_DIR / f"{path.stem}_{variant}.jpg")

            fields = [path.name, f"{variant:<4}"]
            matched = False
            if results:
                best = pick(results, expected)
                text = best.ocr.text if best.ocr else ""
                confidence = ocr_confidence(best.ocr) if best.ocr else None
                matched = expected is not None and normalize(text) == expected
                fields += [
                    text or "(no text read)",
                    f"det {best.detection.confidence:.2f}",
                    f"ocr {confidence:.2f}" if confidence is not None else "ocr n/a",
                ]
            else:
                fields.append("NO PLATE FOUND")
            fields.append(f"{elapsed_ms:.0f} ms")
            if len(results) > 1:
                fields.append(f"{len(results)} plates detected")
            if expected is not None:
                fields.append("MATCH" if matched else "MISS")
                runs += 1
                matches += matched
            print(" | ".join(fields))

    if expected is not None:
        print(f"{matches}/{runs} read correctly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
