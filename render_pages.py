"""Скановані PDF -> PNG для читання сторінок зоровою моделлю.

Кроки для кожної сторінки:
  1. рендер у сірі тони (за замовчуванням 200 dpi);
  2. вирівнювання нахилу (deskew) по нотних лінійках;
  3. для розворотів (2 сторінки на аркуші) - розріз по найдовшому
     порожньому вертикальному проміжку біля центру, щоб не різати літери;
  4. підсилення контрасту (CLAHE) і легке шумозаглушення.

Імена файлів: p0012.png (одна сторінка) або p0012l.png / p0012r.png
(ліва / права половина розвороту) - саме їх очікує merge_songs.py.

Запуск:
    python render_pages.py --book psalmy --out work/psalmy_png
    python render_pages.py --book himny --out work/himny_png --pages 12-20
"""
import argparse
import glob
import os
import re

import cv2
import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))

BOOKS = {
    # Великий скан розбитий на частини: ..._part01_p001-064.pdf тощо.
    "psalmy": dict(pattern="Psalmy-Sions-ASD_part*_p*.pdf", two_up=False),
    "himny": dict(pattern="himny-nadezhdy-asd-1997.pdf", two_up=True),
}


def sources(pattern):
    """Список (перша_сторінка_книги, шлях) у порядку сторінок."""
    out = []
    for path in glob.glob(os.path.join(HERE, pattern)):
        m = re.search(r"_p(\d+)-\d+\.pdf$", path)
        out.append((int(m[1]) if m else 1, path))
    if not out:
        raise FileNotFoundError(f"Не знайдено PDF за шаблоном {pattern!r} у {HERE}")
    return sorted(out)


def render(page, dpi):
    pm = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width).copy()


def deskew(gray):
    """Знайти кут нахилу по горизонтальних нотних лінійках і випрямити."""
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1)))
    lines = cv2.HoughLinesP(horiz, 1, np.pi / 1440, 200,
                            minLineLength=gray.shape[1] // 5, maxLineGap=12)
    if lines is None:
        return gray
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1))
              for x1, y1, x2, y2 in lines.reshape(-1, 4) if x2 != x1]
    angles = [a for a in angles if abs(a) < 5]
    if not angles:
        return gray
    angle = float(np.median(angles))
    if abs(angle) < 0.05:
        return gray
    h, w = gray.shape
    rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, rot, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def split_spread(gray):
    """Розрізати розворот по середині найдовшого порожнього проміжку.

    Розріз точно по центру іноді відтинає краї літер, тому шукаємо
    в центральній смузі найширшу серію колонок без чорнила."""
    h, w = gray.shape
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    ink = bw[int(h * 0.05):int(h * 0.95)].sum(axis=0) / 255
    lo, hi = int(w * 0.35), int(w * 0.65)
    empty = ink[lo:hi] <= max(2, h * 0.002)
    best, start = (0, lo + (hi - lo) // 2, lo + (hi - lo) // 2), None
    for i, flag in enumerate(np.append(empty, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start > best[0]:
                best = (i - start, lo + start, lo + i)
            start = None
    cut = (best[1] + best[2]) // 2
    if best[0] == 0:
        cut = lo + int(np.argmin(ink[lo:hi]))
    return gray[:, :cut], gray[:, cut:]


def enhance(gray):
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7,
                                    searchWindowSize=21)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=BOOKS, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pages", default="", help="номери сторінок книги, напр. 12-20")
    ap.add_argument("--no-enhance", action="store_true")
    a = ap.parse_args()
    cfg = BOOKS[a.book]
    first, last = (int(x) for x in a.pages.split("-")) if a.pages else (1, 10**6)
    os.makedirs(a.out, exist_ok=True)
    count = 0
    for offset, path in sources(cfg["pattern"]):
        with pymupdf.open(path) as doc:
            for i in range(doc.page_count):
                num = offset + i
                if not first <= num <= last:
                    continue
                gray = deskew(render(doc[i], a.dpi))
                parts = (zip("lr", split_spread(gray)) if cfg["two_up"]
                         else [("", gray)])
                for side, img in parts:
                    if not a.no_enhance:
                        img = enhance(img)
                    cv2.imwrite(os.path.join(a.out, f"p{num:04d}{side}.png"), img)
                    count += 1
    print(f"{a.book}: {count} PNG -> {a.out}")


if __name__ == "__main__":
    main()
