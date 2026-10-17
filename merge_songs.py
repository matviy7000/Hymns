"""Merge visually transcribed song batches without losing repeated lyrics."""
import argparse
import copy
import json
import re
from pathlib import Path


def page_key(src):
    match = re.fullmatch(r"p(\d+)([lr]?)\.png", str(src or ""))
    if not match:
        raise ValueError(f"Invalid source image: {src!r}")
    return int(match[1]), {"": 0, "l": 0, "r": 1}[match[2]]


def load_all(folder, prefix, last_batch=None):
    items, pages, legacy = [], [], []
    files = sorted(Path(folder).glob(prefix + "_b*.json"))
    if last_batch is not None:
        files = [path for path in files if int(re.search(r"_b(\d+)", path.name)[1]) <= last_batch]
    if not files:
        raise ValueError(f"No batches found for {prefix!r}")
    for path in files:
        with path.open(encoding="utf-8-sig") as handle:
            data = json.load(handle)
        if isinstance(data, dict) and "songs" in data:
            batch_pages = data.get("pages", [])
            for page in batch_pages:
                page_key(page)
                if page in pages:
                    raise ValueError(f"Image transcribed in multiple batches: {page}")
                pages.append(page)
            data = data["songs"]
        elif isinstance(data, list):
            # The first ten saved batches predate the explicit page manifest.
            legacy.append(path.name)
        else:
            raise ValueError(f"Invalid batch format: {path}")
        if not isinstance(data, list):
            raise ValueError(f"songs must be a list: {path}")
        for item in data:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid song record: {path}")
            page_key(item.get("src"))
            item["_file"] = path.name
            items.append(item)
    items.sort(key=lambda item: page_key(item["src"]))
    return items, sorted(pages, key=page_key), legacy


def norm_label(label):
    return (label or "").strip().upper().rstrip(":")


def merge_blocks(dst, src_blocks):
    by_label = {norm_label(block.get("label")): block for block in dst}
    for block in src_blocks:
        label = norm_label(block.get("label"))
        lines = []
        for line in block.get("lines", []):
            line = line.strip()
            if not line:
                continue
            if lines and re.search(r"[А-Яа-яЁёІіЇїЄєҐґ]-$", lines[-1]):
                lines[-1] = lines[-1][:-1] + line
            else:
                lines.append(line)
        if not lines:
            continue
        if label and label in by_label:
            existing = by_label[label]["lines"]
            if existing and re.search(r"[А-Яа-яЁёІіЇїЄєҐґ]-$", existing[-1]):
                existing[-1] = existing[-1][:-1] + lines.pop(0)
            elif existing and block.get("join_previous"):
                existing[-1] += " " + lines.pop(0)
            # Repeated lines are music, not duplicates to discard.
            existing.extend(lines)
        else:
            new_block = {"label": label, "lines": lines}
            dst.append(new_block)
            by_label[label] = new_block


def merge(items):
    songs, by_num, seen = [], {}, {}
    for item in items:
        num = str(item.get("num", "")).strip().rstrip(".")
        record_key = item["src"], num
        content = {key: value for key, value in item.items() if key != "_file"}
        if record_key in seen:
            if seen[record_key] != content:
                raise ValueError(f"Conflicting duplicate: song {num}, {item['src']}")
            continue
        seen[record_key] = content
        if num == "?":
            if not songs or songs[-1]["complete"]:
                raise ValueError(f"Continuation without an unfinished song: {item['src']}")
            song = songs[-1]
            merge_blocks(song["blocks"], item.get("blocks", []))
            song["complete"] = bool(item.get("complete", True))
            song["sources"].extend(item.get("sources") or [item["src"]])
            continue
        if not num.isdigit():
            raise ValueError(f"Invalid song number: {num!r}, {item['src']}")
        if num in by_num:
            song = by_num[num]
            if song["complete"]:
                raise ValueError(f"Repeated complete song {num}: {item['src']}")
            if song["title"] != (item.get("title") or "").strip():
                raise ValueError(f"Conflicting titles for song {num}")
            merge_blocks(song["blocks"], item.get("blocks", []))
            song["complete"] = bool(item.get("complete", True))
            song["sources"].extend(item.get("sources") or [item["src"]])
            continue
        song = {
            "num": num,
            "title": (item.get("title") or "").strip(),
            "complete": bool(item.get("complete", True)),
            "src": item["src"],
            "sources": list(item.get("sources") or [item["src"]]),
            "blocks": [],
        }
        merge_blocks(song["blocks"], copy.deepcopy(item.get("blocks", [])))
        songs.append(song)
        by_num[num] = song
    for song in songs:
        song["sources"] = sorted(set(song["sources"]), key=page_key)
    return sorted(songs, key=lambda song: int(song["num"]))


def audit(songs, expected_count=None):
    nums = {int(song["num"]) for song in songs}
    last = expected_count if expected_count is not None else max(nums, default=0)
    return {
        "songs": len(songs),
        "expected_songs": expected_count,
        "missing": sorted(set(range(1, last + 1)) - nums),
        "unexpected": sorted(num for num in nums if num < 1 or num > last),
        "incomplete": [song["num"] for song in songs if not song["complete"]],
        "empty": [song["num"] for song in songs if not song["title"] or not song["blocks"]],
        "invalid_labels": [song["num"] for song in songs
                           if any(not re.fullmatch(r"(?:[1-9]\d* КУПЛЕТ|ПРИПЕВ(?: [1-9]\d*)?)",
                                  block["label"]) for block in song["blocks"])],
        "uncertain": [song["num"] for song in songs
                      if "[?]" in song["title"] or any("[?]" in line
                      for block in song["blocks"] for line in block["lines"])],
        "broken_words": [song["num"] for song in songs
                         if any(re.search(r"[А-Яа-яЁёІіЇїЄєҐґ]-$", line)
                                for block in song["blocks"] for line in block["lines"])],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--last-batch", type=int)
    parser.add_argument("--report")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    items, pages, legacy = load_all(args.dir, args.prefix, args.last_batch)
    songs = merge(items)
    report = audit(songs, args.expected_count)
    report.update(pages=pages, legacy_batches_without_page_manifest=legacy)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    if args.strict and any(report[key] for key in
                           ("missing", "unexpected", "incomplete", "empty", "invalid_labels", "uncertain", "broken_words")):
        raise ValueError("Song audit failed; final output was not written")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(songs, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
