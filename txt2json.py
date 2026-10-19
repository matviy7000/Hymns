"""Компактна розмітка прочитаної пачки -> json/<prefix>_bNNN.json (формат PROMPT_PAGE_READ.txt).

Розмітка:
  @ p0012.png          поточне зображення (кожне прочитане зображення має свій @)
  # 11 | Назва         нова пісня;  "# ?" - продовження пісні з попередньої сторінки
  - 1 КУПЛЕТ           новий блок ("-" без тексту - блок без позначки)
  -+ 2 КУПЛЕТ          блок, перший рядок якого продовжує обірваний рядок (join_previous)
  %                    поточна пісня не закінчилась на цій сторінці (complete=false)
  інші рядки           рядки тексту поточного блоку

Запуск: python txt2json.py batch.txt out.json
"""
import json
import re
import sys


def convert(text):
    pages, songs, song, block = [], [], None, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("@"):
            src = line[1:].strip()
            assert src not in pages, src
            pages.append(src)
            song = block = None
        elif line.startswith("#"):
            num, _, title = line[1:].partition("|")
            song = {"src": pages[-1], "num": num.strip(), "title": title.strip(),
                    "complete": True, "blocks": []}
            if song["num"] == "?":
                del song["title"]
            songs.append(song)
            block = None
        elif line.startswith("-+") or line == "-" or line.startswith("- "):
            join = line.startswith("-+")
            block = {"label": line[2:].strip(), "lines": []}
            if join:
                block["join_previous"] = True
            song["blocks"].append(block)
        elif line == "%":
            song["complete"] = False
        else:
            block["lines"].append(line)
    for s in songs:
        assert s["blocks"] and all(b["lines"] for b in s["blocks"]), s
        for b in s["blocks"]:
            for line in b["lines"] + [s.get("title", "")]:
                assert not re.search(r"[A-HJ-UWYZa-z]", line), f"латиниця: {line!r}"
    return {"pages": pages, "songs": songs}


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        data = convert(f.read())
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(sys.argv[2], len(data["pages"]), "pages,", len(data["songs"]), "records:",
          " ".join(s["num"] + ("" if s["complete"] else "…") for s in data["songs"]))
