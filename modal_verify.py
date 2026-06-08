r"""
modal_verify.py  —  count the dataset files on the chexpert-data volume so you
can match them against the local count.

Run:
    modal run modal_verify.py
    modal run modal_verify.py --subdir CheXpert-v1.0-small

Compare its `.jpg` total with the local count (PowerShell):
    (Get-ChildItem 'D:\CheXpert-v1.0-small' -Recurse -File -Filter *.jpg).Count
"""

import modal

VOLUME = "chexpert-data"
MOUNT = "/data"

app = modal.App("chexpert-verify")
image = modal.Image.debian_slim(python_version="3.11")
vol = modal.Volume.from_name(VOLUME)


@app.function(image=image, volumes={MOUNT: vol}, timeout=2 * 3600)
def count_files(subdir: str):
    import os

    vol.reload()                       # see the latest committed state
    root = f"{MOUNT}/{subdir}"
    print(f"[verify] volume='{VOLUME}'  root={root}")
    if not os.path.isdir(root):
        print(f"[verify] ⚠️ '{root}' not found. {MOUNT} contains: {sorted(os.listdir(MOUNT))}")
        return

    total, jpg = 0, 0
    for _dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            total += 1
            if f.lower().endswith(".jpg"):
                jpg += 1

    # per top-level split (train / valid / test), jpg only
    per = {}
    for split in ("train", "valid", "test"):
        sp = f"{root}/{split}"
        c = 0
        if os.path.isdir(sp):
            for _dp, _dn, fs in os.walk(sp):
                c += sum(1 for f in fs if f.lower().endswith(".jpg"))
        per[split] = c

    print(f"[verify] total files = {total}")
    print(f"[verify] .jpg images = {jpg}")
    print(f"[verify] per split (.jpg): {per}")
    # also count the split CSVs if present
    splits_dir = f"{MOUNT}/splits"
    if os.path.isdir(splits_dir):
        print(f"[verify] /splits contains: {sorted(os.listdir(splits_dir))}")


@app.local_entrypoint()
def main(subdir: str = "CheXpert-v1.0-small"):
    count_files.remote(subdir)
