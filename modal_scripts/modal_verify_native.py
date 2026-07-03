r"""
modal_verify_native.py  —  inspect the native-resolution CheXpert download on the
chexpert-native-data volume: list every entry with its size, flag azcopy's orphaned
`.azDownload-*` partials, and report which expected files are present/missing.

Run (report only):
    modal run modal_verify_native.py

Delete the orphaned `.azDownload-*` partials to reclaim space (destructive):
    modal run modal_verify_native.py --clean
"""

import modal

VOLUME = "chexpert-native-data"
MOUNT = "/data_native"
CONTAINER = "chexpertchestxrays-u20210408"   # azcopy nests the blob container here

# the clean (non-temp) files expected after a successful azcopy pull
EXPECTED = [
    "CheXpert-v1.0 batch 1 (validate & csv).zip",
    "CheXpert-v1.0 batch 2 (train 1).zip",
    "CheXpert-v1.0 batch 3 (train 2).zip",
    "CheXpert-v1.0 batch 4 (train 3).zip",
    "CHEXPERT DEMO.xlsx",
    "README.md",
    "train_cheXbert.csv",
    "train_visualCheXbert.csv",
]

app = modal.App("chexpert-verify-native")
image = modal.Image.debian_slim(python_version="3.11")
vol = modal.Volume.from_name(VOLUME)


def _human(n: int) -> str:
    x = float(n)
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if x < 1024 or u == "TiB":
            return f"{x:,.1f} {u}"
        x /= 1024


@app.function(image=image, volumes={MOUNT: vol}, timeout=2 * 3600)
def verify(clean: bool):
    import os

    vol.reload()
    # the files may sit directly under MOUNT or under the container subfolder
    root = f"{MOUNT}/{CONTAINER}" if os.path.isdir(f"{MOUNT}/{CONTAINER}") else MOUNT
    print(f"[verify] volume='{VOLUME}'  root={root}")
    if not os.path.isdir(root):
        print(f"[verify] ⚠️ '{root}' not found. {MOUNT} contains: {sorted(os.listdir(MOUNT))}")
        return

    entries = sorted(os.listdir(root))
    orphans, clean_files, total = [], {}, 0
    print(f"[verify] {len(entries)} entries:")
    for name in entries:
        p = f"{root}/{name}"
        size = os.path.getsize(p) if os.path.isfile(p) else sum(
            os.path.getsize(os.path.join(dp, f))
            for dp, _dn, fs in os.walk(p) for f in fs)
        total += size
        is_orphan = name.startswith(".azDownload-")
        if is_orphan:
            orphans.append((name, size))
        else:
            clean_files[name] = size
        tag = "  ⛔ ORPHAN PARTIAL" if is_orphan else ""
        print(f"    {_human(size):>12}   {name}{tag}")

    orphan_bytes = sum(s for _n, s in orphans)
    print(f"\n[verify] total on disk : {_human(total)}")
    print(f"[verify] orphan partials: {len(orphans)}  ({_human(orphan_bytes)} reclaimable)")

    missing = [f for f in EXPECTED if f not in clean_files]
    present = [f for f in EXPECTED if f in clean_files]
    print(f"[verify] expected present: {len(present)}/{len(EXPECTED)}")
    if missing:
        print(f"[verify] ⚠️ MISSING: {missing}")
    else:
        print("[verify] ✅ all expected clean files present")

    if clean and orphans:
        print(f"\n[clean] deleting {len(orphans)} orphaned partial(s) ...")
        for name, size in orphans:
            os.remove(f"{root}/{name}")
            print(f"[clean] removed {name}  ({_human(size)})")
        vol.commit()
        print(f"[clean] committed — reclaimed {_human(orphan_bytes)} ✅")
    elif clean:
        print("\n[clean] nothing to delete (no orphans).")
    elif orphans:
        print("\n[verify] re-run with --clean to delete the orphaned partials.")


@app.local_entrypoint()
def main(clean: bool = False):
    verify.remote(clean)
