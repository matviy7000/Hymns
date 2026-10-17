"""
Пайплайн: скан нотного збірника -> PDF лише зі словами (одна пісня = одна сторінка).

Етапи:
  1. рендер сторінки з PDF у зображення (300 dpi)
  2. deskew (вирівнювання перекосу за нотними лінійками)
  3. розрізання розкладок "2 аркуші на сторінку" (Гимны надежды)
  4. OCR зі склеюванням складів у цілі рядки (EasyOCR / CRAFT)
  5. фільтр нотного сміття за висотою боксу, впевненістю і часткою кирилиці
  6. визначення заголовків (номер + назва), збирання куплетів і приспіву
  7. де-гіфенізація складів -> нормальні слова
  8. генерація PDF у стилі зразка

Запуск:
    python hymn_pipeline.py --book psalmy --pages 118-124 --out demo.pdf
"""
import argparse
import json
import os
import re

import cv2
import numpy as np
import pymupdf

DPI = 300
SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Нова папка")

BOOKS = {
    "psalmy": dict(file="Psalmy-Sions-ASD", title="Псалми Сіону", two_up=False),
    "himny": dict(file="himny-nadezhdy-asd-1997", title="Гимны надежды",
                  two_up=True),
}

CYR = re.compile(r"[А-Яа-яЁёІіЇїЄєҐґ]")
NUM_ONLY = re.compile(r"^\s*(\d{1,3})\s*[.,]?\s*$")
CHORUS = re.compile(r"при[пr][еe]в", re.I)

# межі висоти рядка тексту в пікселях при 300 dpi
LINE_H = (28, 85)
# висота заголовка
HEAD_H = (70, 140)


# ---------------------------------------------------------------- рендер


def render_page(doc, idx, dpi=DPI):
    pm = doc[idx].get_pixmap(dpi=dpi)
    img = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    if pm.n >= 3:
        return cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    return img[:, :, 0].copy()


def deskew(gray):
    bw = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 1))
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    lines = cv2.HoughLinesP(horiz, 1, np.pi / 1440, 200,
                            minLineLength=gray.shape[1] // 5, maxLineGap=12)
    if lines is None:
        return gray
    angles = [
        np.degrees(np.arctan2(y2 - y1, x2 - x1))
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4)
        if x2 != x1
    ]
    angles = [a for a in angles if abs(a) < 5]
    if not angles:
        return gray
    ang = float(np.median(angles))
    if abs(ang) < 0.05:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def split_spread(gray):
    """Розрізати розкладку 2-up по вертикальному проміжку в центрі."""
    h, w = gray.shape
    bw = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    col = bw[int(h * 0.1):int(h * 0.9)].sum(axis=0)
    lo, hi = int(w * 0.44), int(w * 0.56)
    g = lo + int(np.argmin(col[lo:hi]))
    return [gray[:, :g], gray[:, g:]]


# ------------------------------------------------------------------ OCR

OCR_OPTS = dict(detail=1, paragraph=False, width_ths=4.0, ycenter_ths=0.35,
                height_ths=0.6, slope_ths=0.25, mag_ratio=2.0,
                text_threshold=0.6, low_text=0.35)


def ocr(reader, gray):
    out = []
    for box, txt, conf in reader.readtext(gray, **OCR_OPTS):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        out.append(dict(y0=int(min(ys)), y1=int(max(ys)), x0=int(min(xs)),
                        x1=int(max(xs)), h=int(max(ys) - min(ys)),
                        text=txt.strip(), conf=float(conf)))
    out.sort(key=lambda b: (b["y0"], b["x0"]))
    return out


def cyr_ratio(s):
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(CYR.match(c)) for c in letters) / len(letters)


def is_lyric(b):
    if not (LINE_H[0] <= b["h"] <= LINE_H[1]):
        return False
    if b["conf"] < 0.25:
        return False
    if len(re.sub(r"[^\wА-Яа-яЁёІіЇїЄєҐґ]", "", b["text"])) < 3:
        return False
    return cyr_ratio(b["text"]) >= 0.6


# ------------------------------------------------- нормалізація тексту

def dehyphenate(s):
    """Склеїти склади, рознесені під нотами, у цілі слова."""
    t = s
    # OCR читає дефіс між складами як . , ~ ‚ або просто -
    t = re.sub(r"(?<=[А-Яа-яЁёІіЇїЄєҐґ])\s*[-–—.,~‚]\s*"
               r"(?=[А-Яа-яЁёІіЇїЄєҐґ])", "\x00", t)
    # подвійний+ пробіл = справжня межа слова
    t = re.sub(r"[ \t]{2,}", "\x01", t)
    # одинарний пробіл між складами теж часто розрив складу
    parts = t.split("\x01")
    fixed = []
    for p in parts:
        p = p.replace("\x00", "")
        fixed.append(p)
    t = " ".join(x for x in fixed if x)
    # нормалізувати регістр усередині слова
    def fix_case(m):
        w = m.group(0)
        if w.isupper() and len(w) > 3:
            return w
        return w[0] + w[1:].lower()
    t = re.sub(r"[А-Яа-яЁёІіЇїЄєҐґ]+", fix_case, t)
    return re.sub(r"\s+", " ", t).strip()


def strip_lead_num(s):
    m = re.match(r"^\s*(\d{1,2})[.)]?\s+(.*)$", s)
    if m:
        return int(m.group(1)), m.group(2)
    return None, s


# --------------------------------------------------- структура пісень

def parse_page(boxes, page_label):
    """Знайти заголовки та рядки тексту на одній (пів)сторінці."""
    heads = []
    for b in boxes:
        m = NUM_ONLY.match(b["text"])
        if m and b["h"] >= HEAD_H[0] and b["conf"] > 0.4:
            # назва - найширший кириличний бокс на тій самій висоті
            cand = [
                o for o in boxes
                if abs(o["y0"] - b["y0"]) < b["h"] * 1.2
                and o is not b and cyr_ratio(o["text"]) > 0.6
                and len(o["text"]) > 3
            ]
            cand.sort(key=lambda o: -(o["x1"] - o["x0"]))
            title = cand[0]["text"].strip(" .,:;") if cand else ""
            heads.append(dict(num=int(m.group(1)), title=title, y=b["y0"]))
    lyr = [b for b in boxes if is_lyric(b)]
    return heads, lyr, page_label


def group_rows(lines, gap_factor=1.45):
    """Згрупувати рядки у 'смуги' (між станами), а в смузі - у ряди куплетів."""
    if not lines:
        return []
    med_h = float(np.median([b["h"] for b in lines])) or 40.0
    bands, cur = [], [lines[0]]
    for prev, b in zip(lines, lines[1:]):
        if b["y0"] - prev["y0"] > med_h * 3.2:
            bands.append(cur)
            cur = [b]
        else:
            cur.append(b)
    bands.append(cur)

    out = []
    for band in bands:
        band = sorted(band, key=lambda b: (b["y0"], b["x0"]))
        rows, cur = [], [band[0]]
        for prev, b in zip(band, band[1:]):
            if b["y0"] - prev["y0"] > med_h * gap_factor:
                rows.append(cur)
                cur = [b]
            else:
                cur.append(b)
        rows.append(cur)
        merged = []
        for r in rows:
            r = sorted(r, key=lambda b: b["x0"])
            merged.append("  ".join(x["text"] for x in r))
        out.append(merged)
    return out


def assemble(heads, lines):
    """Побудувати куплети: ряд k кожної смуги -> куплет k."""
    bands = group_rows(lines)
    chorus_at = None
    for i, band in enumerate(bands):
        if any(CHORUS.search(r) for r in band):
            chorus_at = i
            break
    verses, chorus = {}, []
    for i, band in enumerate(bands):
        for k, raw in enumerate(band):
            txt = dehyphenate(CHORUS.sub("", raw))
            if not txt:
                continue
            n, rest = strip_lead_num(txt)
            if n:
                txt = rest
            if chorus_at is not None and i >= chorus_at:
                chorus.append(txt)
            else:
                verses.setdefault((n or k + 1), []).append(txt)
    return verses, chorus


# -------------------------------------------------------------- PDF

def build_pdf(songs, out_path, book_title):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    F = r"C:\Windows\Fonts"
    pdfmetrics.registerFont(TTFont("S", os.path.join(F, "arial.ttf")))
    pdfmetrics.registerFont(TTFont("SB", os.path.join(F, "arialbd.ttf")))
    pdfmetrics.registerFont(TTFont("SBI", os.path.join(F, "arialbi.ttf")))

    W, H = A4
    L, R, T, B = 20 * mm, 18 * mm, 22 * mm, 16 * mm
    c = canvas.Canvas(out_path, pagesize=A4)
    c.setTitle(book_title)

    for s in songs:
        y = H - T
        c.setFont("SBI", 17)
        c.drawString(L, y, f"{s['num']}.   {s['title']}")
        y -= 13 * mm

        def block(label, rows, indent=0):
            nonlocal y
            if not rows:
                return
            c.setFont("SB", 9.5)
            c.drawString(L + indent, y, label)
            y -= 6.4 * mm
            c.setFont("S", 11.5)
            for r in rows:
                for seg in wrap(r, W - L - R - indent):
                    c.drawString(L + indent, y, seg)
                    y -= 5.9 * mm
            y -= 4.5 * mm

        def wrap(text, maxw):
            words, line, res = text.split(), "", []
            for w_ in words:
                probe = (line + " " + w_).strip()
                if c.stringWidth(probe, "S", 11.5) <= maxw:
                    line = probe
                else:
                    if line:
                        res.append(line)
                    line = w_
            if line:
                res.append(line)
            return res or [""]

        for k in sorted(s["verses"]):
            block(f"{k} КУПЛЕТ:", s["verses"][k])
            if k == 1 and s["chorus"]:
                block("ПРИПЕВ:", s["chorus"], indent=8 * mm)
        c.showPage()
    c.save()
    return out_path


# -------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--book", choices=BOOKS, required=True)
    ap.add_argument("--pages", default="", help="напр. 118-124")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-json", default="")
    a = ap.parse_args()

    import easyocr
    cfg = BOOKS[a.book]
    doc = pymupdf.open(os.path.join(SRC, cfg["file"] + ".pdf"))
    if a.pages:
        p0, p1 = (int(x) for x in a.pages.split("-"))
        rng = range(p0 - 1, min(p1, doc.page_count))
    else:
        rng = range(doc.page_count)

    reader = easyocr.Reader(["ru"], gpu=False, verbose=False)
    pages = []
    for i in rng:
        gray = deskew(render_page(doc, i))
        halves = split_spread(gray) if cfg["two_up"] else [gray]
        for j, half in enumerate(halves):
            heads, lyr, _ = parse_page(ocr(reader, half), f"{i+1}.{j}")
            pages.append((heads, lyr))
            print(f"  стор {i+1}.{j}: заголовків={len(heads)} "
                  f"рядків={len(lyr)}"
                  + (f"  -> {heads[0]['num']} {heads[0]['title']!r}"
                     if heads else ""))
    doc.close()

    # розбити потік на пісні за заголовками
    songs, cur = [], None
    for heads, lyr in pages:
        if heads:
            for h in heads:
                if cur:
                    songs.append(cur)
                cur = dict(num=h["num"], title=h["title"], lines=[])
            if cur:
                cur["lines"] += lyr
        elif cur:
            cur["lines"] += lyr
    if cur:
        songs.append(cur)

    final = []
    for s in songs:
        v, ch = assemble([], s["lines"])
        if v or ch:
            final.append(dict(num=s["num"], title=s["title"],
                              verses=v, chorus=ch))

    if a.dump_json:
        with open(a.dump_json, "w", encoding="utf-8") as f:
            json.dump(final, f, ensure_ascii=False, indent=1,
                      default=str)
    build_pdf(final, a.out, BOOKS[a.book]["title"])
    print(f"\nпісень: {len(final)}  ->  {a.out}")


if __name__ == "__main__":
    main()
