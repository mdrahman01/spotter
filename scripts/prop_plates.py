"""Prop licence plates for filming, so the real plate never appears on video.

    .venv/bin/python scripts/prop_plates.py [--pdf props/prop_plates.pdf] [--png-dir data/props]

Two fictional plates, one per US Letter page in landscape, each 10.5 x 5.25 in:
plain white, black characters, a thin border, "SPOTTER DEMO" small at the top
and "PROP - NOT A LICENSE PLATE" small at the bottom. No state name and no
official marks. The characters avoid the ones the plate reader confuses
(O 0 I 1 Q B 8 S 5 Z 2 G 6). Print the PDF at 100% and cut along the border.
The PNG of each plate (the plate alone, at 300 dpi) is written for the reader
test under data/, which stays out of git.
"""

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
PLATES = ("RENT 743", "MAYR 473")  # both read 100% in the reader test; PARK479 dropped to 84% at 70 px
CONFUSABLE = set("O0I1QB8S5Z2G6")
DPI = 300
PAGE_IN = (11.0, 8.5)  # US Letter, landscape
PLATE_IN = (10.5, 5.25)
FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
TOP_LABEL = "SPOTTER DEMO"
BOTTOM_LABEL = "PROP - NOT A LICENSE PLATE"


def px(inches: float) -> int:
    return round(inches * DPI)


def font(size_px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size_px)


def fitted_font(text: str, max_width: int, max_height: int, tracking: float) -> ImageFont.FreeTypeFont:
    """The largest font whose characters, spaced by `tracking` of their size, fit the box."""
    size = max_height
    while size > 10:
        f = font(size)
        width = sum(f.getlength(ch) for ch in text) + tracking * size * (len(text) - 1)
        top, bottom = f.getbbox("H")[1], f.getbbox("H")[3]
        if width <= max_width and bottom - top <= max_height:
            return f
        size -= 8
    return font(size)


def draw_centered(draw: ImageDraw.ImageDraw, text: str, f: ImageFont.FreeTypeFont, cx: int, cy: int, tracking: float = 0.0) -> None:
    """Draw `text` with its visual centre at (cx, cy), with extra space between characters."""
    gap = tracking * f.size
    widths = [f.getlength(ch) for ch in text]
    total = sum(widths) + gap * (len(text) - 1)
    bbox = f.getbbox("H")
    y = cy - (bbox[1] + bbox[3]) / 2
    x = cx - total / 2
    for ch, w in zip(text, widths):
        draw.text((x, y), ch, font=f, fill="black")
        x += w + gap


def render_plate(text: str) -> Image.Image:
    """The plate alone: white, black characters, a thin border and the two small labels."""
    width, height = px(PLATE_IN[0]), px(PLATE_IN[1])
    plate = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(plate)
    inset, stroke = px(0.08), px(0.03)
    draw.rounded_rectangle((inset, inset, width - inset, height - inset), radius=px(0.25), outline="black", width=stroke)
    label = font(px(0.30))
    draw_centered(draw, TOP_LABEL, label, width // 2, px(0.55))
    draw_centered(draw, BOTTOM_LABEL, label, width // 2, height - px(0.55))
    main = fitted_font(text, width - px(1.2), px(2.6), tracking=0.06)
    draw_centered(draw, text, main, width // 2, height // 2 + px(0.05), tracking=0.06)
    return plate


def page_with(plate: Image.Image) -> Image.Image:
    page = Image.new("RGB", (px(PAGE_IN[0]), px(PAGE_IN[1])), "white")
    page.paste(plate, ((page.width - plate.width) // 2, (page.height - plate.height) // 2))
    return page


def main() -> int:
    parser = argparse.ArgumentParser(description="Make the prop plates PDF and the plate PNGs.")
    parser.add_argument("--pdf", default=str(REPO_ROOT / "props" / "prop_plates.pdf"))
    parser.add_argument("--png-dir", default=str(REPO_ROOT / "data" / "props"))
    parser.add_argument("--plates", nargs="+", default=list(PLATES))
    args = parser.parse_args()
    for text in args.plates:
        bad = sorted(set(text.replace(" ", "")) & CONFUSABLE)
        if bad:
            print(f"{text}: uses characters the reader confuses: {''.join(bad)}", file=sys.stderr)
            return 2
    pdf, png_dir = Path(args.pdf), Path(args.png_dir)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for text in args.plates:
        plate = render_plate(text)
        plate.save(png_dir / f"{text.replace(' ', '')}.png")
        pages.append(page_with(plate))
    pages[0].save(pdf, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
    print(f"{pdf}: {len(pages)} pages, {PAGE_IN[0]:g} x {PAGE_IN[1]:g} in, plate {PLATE_IN[0]:g} x {PLATE_IN[1]:g} in")
    print(f"plate PNGs in {png_dir}: {', '.join(t.replace(' ', '') + '.png' for t in args.plates)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
