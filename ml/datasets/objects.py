"""Turn a downloaded YOLO-format object dataset (Roboflow export, spec D5) into SAGA classes.

Public exam/phone/paper datasets all use their own class names — `mobile`, `cell phone`, `cheat
sheet`, `paper note`, `headphone`… This maps them onto the fixed SAGA class list so that one
detector can be trained from several exports, and so the class names the rules check
(`obj_class in ['phone', ...]`) never depend on which dataset was used.

Labels are rewritten (class index remapped, unmapped classes dropped); images are NOT copied —
the new data.yaml points at the original image folders, which stay outside the repo.

    from ml.datasets.objects import prepare
    data_yaml = prepare(r"C:\\SAGA_data\\exam_objects")
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import yaml

from edge.perception.objects import CLASSES

ALIASES = {
    "phone": ["phone", "mobile", "mobile phone", "cell phone", "cellphone", "smartphone", "handphone",
              "using phone", "phone use", "mobile_phone", "cell-phone"],
    "chit": ["chit", "chits", "paper", "paper note", "note", "notes", "cheat sheet", "cheatsheet",
             "cheat-sheet", "cheat slip", "cheatslip", "slip", "chit paper", "micro script",
             "cheating paper", "cheat paper", "note passing", "exchanging paper"],
    "earphone": ["earphone", "earphones", "earbud", "earbuds", "headphone", "headphones", "airpods",
                 "bluetooth earphone"],
    "smartwatch": ["smartwatch", "smart watch", "watch", "wrist watch", "wristwatch"],
    "calculator": ["calculator", "calc", "scientific calculator"],
    "book": ["book", "books", "notebook", "textbook", "register"],
    "bottle": ["bottle", "water bottle", "waterbottle"],
    "pen": ["pen", "pencil", "stationery"],
}
LOOKUP = {alias.lower(): saga for saga, aliases in ALIASES.items() for alias in aliases}


def map_name(name: str) -> str | None:
    """Dataset class name -> SAGA class, or None if it is not something we reason about."""
    n = name.strip().lower().replace("_", " ")
    return LOOKUP.get(n) or (n if n in CLASSES else None)


def all_classes(roots) -> list[str]:
    """The SAGA classes present across several exports, in the canonical order."""
    found = set()
    for r in roots:
        data = yaml.safe_load((Path(r) / "data.yaml").read_text(encoding="utf-8"))
        found |= {c for c in (map_name(n) for n in _names(data)) if c}
    return sorted(found, key=CLASSES.index)


def _prepare_many(roots: list[Path], out: str | Path | None, classes: list[str] | None) -> Path:
    keep = classes or all_classes(roots)
    out = Path(out or roots[0].parent / "objects_merged_saga")
    splits: dict[str, list[str]] = {}
    for r in roots:
        sub = prepare(r, out / r.name, keep)      # same class list for every dataset
        cfg = yaml.safe_load(sub.read_text(encoding="utf-8"))
        for split in ("train", "val", "test"):
            if cfg.get(split):
                splits.setdefault(split, []).append(cfg[split])
    merged = {"path": str(out), "names": {i: c for i, c in enumerate(keep)}, **splits}
    (out / "data.yaml").write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    print(f"merged {len(roots)} datasets -> {out / 'data.yaml'} | classes {keep}")
    return out / "data.yaml"


def _link_images(src: Path, dst: Path) -> Path:
    """Point `dst` at the original image folder without copying it. A symlink needs developer mode
    on Windows, a directory junction does not; copying is the last resort."""
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=True)
        return dst
    except OSError:
        pass
    if subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                      capture_output=True).returncode == 0:
        return dst
    print(f"  no symlink or junction possible — copying {src.name} images into {dst} (one-off)")
    shutil.copytree(src, dst)
    return dst


def _names(data: dict) -> list[str]:
    names = data["names"]
    return [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)


def prepare(roots, out: str | Path | None = None, classes: list[str] | None = None) -> Path:
    """Remap one or more YOLO exports to SAGA classes. Returns the merged data.yaml.

    Several exports can be trained together (one set is strong on phones, another on paper): pass a
    list of folders. Each keeps its own `<out>/<name>/<split>/{images,labels}` pair — filenames
    collide across datasets otherwise — and the data.yaml lists all of them per split, which
    Ultralytics accepts. One shared class list is used throughout, so `chit` means the same thing
    whichever export a box came from. Re-running is cheap and idempotent."""
    roots = [Path(r) for r in ([roots] if isinstance(roots, (str, Path)) else roots)]
    if len(roots) > 1:
        return _prepare_many(roots, out, classes)
    root = roots[0]
    src_yaml = root / "data.yaml"
    if not src_yaml.exists():
        raise FileNotFoundError(f"{src_yaml} not found — point --data at the folder of a YOLO export")
    data = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    src_names = _names(data)
    keep = classes or sorted({c for c in (map_name(n) for n in src_names) if c}, key=CLASSES.index)
    remap = {i: keep.index(m) for i, n in enumerate(src_names) if (m := map_name(n)) in keep}
    if not remap:
        raise ValueError(f"none of {src_names} map to SAGA classes {CLASSES}")

    out = Path(out or root.parent / f"{root.name}_saga")
    splits, dropped, written = {}, 0, 0
    for split in ("train", "val", "test"):
        rel = data.get(split)
        if not rel:
            continue
        img_dir = (root / str(rel).replace("../", "")).resolve()
        if not img_dir.exists():
            continue
        lab_src = img_dir.parent / "labels"
        lab_out = out / split / "labels"      # mirror the <split>/images + <split>/labels layout
        lab_out.mkdir(parents=True, exist_ok=True)
        for txt in lab_src.glob("*.txt"):
            lines = []
            for line in txt.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                if cls in remap:
                    lines.append(" ".join([str(remap[cls])] + parts[1:]))
                else:
                    dropped += 1
            (lab_out / txt.name).write_text("\n".join(lines) + ("\n" if lines else ""))
            written += 1
        # Ultralytics finds labels by swapping /images/ for /labels/ in the image path, so the
        # remapped labels MUST sit next to an images/ folder of their own — pointing at the original
        # images would silently train on the original (differently ordered) class indices.
        splits[split] = str(_link_images(img_dir, out / split / "images"))

    cfg = {"path": str(out), "names": {i: c for i, c in enumerate(keep)}, **splits}
    (out / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"{written} label files -> {out} | classes {keep} | {dropped} boxes of unmapped classes dropped")
    return out / "data.yaml"
