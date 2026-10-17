"""JSON з текстами пісень -> PDF у стилі зразка (одна пісня = одна сторінка).

Схема JSON: список об'єктів
  {"num": "132", "title": "Чудные минуты",
   "blocks": [{"label": "1 КУПЛЕТ", "lines": ["рядок", ...]},
              {"label": "ПРИПЕВ",   "lines": [...]}]}
"""
import os, json, argparse
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

MM = 72.0 / 25.4
FONTS = r"C:\Windows\Fonts"


def reg():
    """Зареєструвати кириличні шрифти Windows."""
    pairs = [("S", "arial.ttf"), ("SB", "arialbd.ttf"),
             ("SBI", "arialbi.ttf"), ("SI", "ariali.ttf")]
    for name, fn in pairs:
        pdfmetrics.registerFont(TTFont(name, os.path.join(FONTS, fn)))


def wrap(text, font, size, width):
    """Перенести рядок по ширині, повертає список рядків."""
    words, out, cur = text.split(), [], ""
    for w in words:
        if pdfmetrics.stringWidth(w, font, size) > width:
            if cur:
                out.append(cur)
                cur = ""
            for char in w:
                if cur and pdfmetrics.stringWidth(cur + char, font, size) > width:
                    out.append(cur)
                    cur = ""
                cur += char
            continue
        t = (cur + " " + w).strip()
        if pdfmetrics.stringWidth(t, font, size) <= width or not cur:
            cur = t
        else:
            out.append(cur)
            cur = w
    if cur:
        out.append(cur)
    return out or [""]


def build(songs, out_path, book_title):
    reg()
    W, H = A4
    L, R, T, B = 20 * MM, 18 * MM, 20 * MM, 16 * MM
    avail = W - L - R
    layouts, seen = [], set()
    for s in songs:
        num = str(s["num"])
        if num in seen or not num.isdigit() or not s.get("complete", True):
            raise ValueError(f"Invalid, duplicate or unfinished song: {num}")
        seen.add(num)
        if not s.get("title") or not any(blk.get("lines") for blk in s.get("blocks", [])):
            raise ValueError(f"Empty song: {num}")
        head = f"{num}.   {s['title']}"
        fitted = False
        # Measure before drawing: a long song must not create a second page.
        for quarter in range(15):
            size = 11.5 - quarter * 0.25
            scale = size / 11.5
            head_size, label_size = max(13, 17 * scale), max(8, 9.5 * scale)
            for compact in (False, True):
                leading = (4.8 if compact else 5.9) * MM * scale
                gap = (2 if compact else 3.4) * MM * scale
                ops = [("SBI", head_size, 0, line, 8.2 * MM * scale)
                       for line in wrap(head, "SBI", head_size, avail)]
                ops.append((None, 0, 0, None, 2.5 * MM * scale))
                for blk in s.get("blocks", []):
                    lines = [x.strip() for x in blk.get("lines", []) if x and x.strip()]
                    if not lines:
                        continue
                    label = (blk.get("label") or "").strip().upper().rstrip(":")
                    if label:
                        ops.append(("SB", label_size, 0, label + ":", 5.6 * MM * scale))
                    for raw in lines:
                        ops.extend(("S", size, 2 * MM, line, leading)
                                   for line in wrap(raw, "S", size, avail - 2 * MM))
                    ops.append((None, 0, 0, None, gap))
                if sum(op[4] for op in ops) <= H - T - B:
                    layouts.append(ops)
                    fitted = True
                    break
            if fitted:
                break
        if not fitted:
            raise ValueError(f"Song {num} cannot fit one A4 page at a readable font size")
    if not layouts:
        raise ValueError("No songs to print")
    c = canvas.Canvas(out_path, pagesize=A4, pageCompression=1)
    c.setTitle(book_title)
    for ops in layouts:
        y = H - T
        for font, size, indent, text, step in ops:
            if text is not None:
                c.setFont(font, size)
                c.drawString(L + indent, y, text)
            y -= step
        c.showPage()
    c.save()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", required=True)
    a = ap.parse_args()
    with open(a.json, encoding="utf-8") as f:
        songs = json.load(f)
    songs.sort(key=lambda s: (int(s["num"]) if str(s["num"]).isdigit() else 10**6,
                              str(s["num"])))
    build(songs, a.out, a.title)
    print(f"пісень: {len(songs)} -> {a.out}")


if __name__ == "__main__":
    main()
