r"""
modal_extract_native.py  —  extract the native-resolution CheXpert zips on the
chexpert-native-data volume, server-side, and merge them into one dataset tree.

The AIMI download landed as 4 ZIP archives under
    /data_native/chexpertchestxrays-u20210408/
        CheXpert-v1.0 batch 1 (validate & csv).zip   (~486 MiB: valid/ + CSVs)
        CheXpert-v1.0 batch 2 (train 1).zip          (~162 GiB: part of train/)
        CheXpert-v1.0 batch 3 (train 2).zip          (~185 GiB: part of train/)
        CheXpert-v1.0 batch 4 (train 3).zip          (~ 91 GiB: part of train/)
All four share the same top-level `CheXpert-v1.0/` prefix, so extracting them into
the SAME target merges them into one tree:  /data_native/CheXpert-v1.0/{train,valid}/...
(-> use remote_data_root=/data_native/CheXpert-v1.0 in a training config).

Extraction uses bsdtar (libarchive; fast, zip64/>4GB safe), unar as a fallback.

The four zips write DISJOINT paths (batch 1 -> valid/ + CSVs; batches 2-4 -> three
disjoint parts of train/), so they are extracted IN PARALLEL — one container per
zip, fanned out with .map. Wall time drops from the SUM of all zips (~438 GiB) to
roughly the LARGEST single zip (~185 GiB). Each container extracts its one zip,
commits (progress survives a restart), then optionally deletes that zip.

Space notes (relevant because they run concurrently):
  * Modal Volume auto-grows — no manual pre-allocation. During a keep-zips run it
    holds the zips (~438 GiB) AND the extracted tree (~438 GiB) at once (~875 GiB
    peak). --delete-zips drops each zip right after it extracts, lowering the peak.
  * ephemeral_disk is PER container (not shared): each of the 4 parallel containers
    gets its own scratch, sized to cover the largest single zip with headroom.

Run (extract only, keep the zips):
    modal run modal_extract_native.py

Extract AND delete each zip right after it extracts OK (keeps peak disk lower;
destructive — only the extracted files remain):
    modal run modal_extract_native.py --delete-zips
"""

import modal

VOLUME = "chexpert-native-data"
MOUNT = "/data_native"
SRC_SUBDIR = "chexpertchestxrays-u20210408"   # where azcopy put the zips
EXPECTED_ROOT = "CheXpert-v1.0"               # the merged top-level folder the zips create

app = modal.App("chexpert-extract-native")
# bsdtar (libarchive) is the primary extractor; unar is the fallback.
image = modal.Image.debian_slim(python_version="3.11").apt_install("libarchive-tools", "unar")
vol = modal.Volume.from_name(VOLUME)


def _human(n: float) -> str:
    for u in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or u == "TiB":
            return f"{n:,.1f} {u}"
        n /= 1024


@app.function(image=image, volumes={MOUNT: vol})
def list_zips():
    """Enumerate the .zip archives on the volume (skip azcopy .azDownload-* partials)
    so the local entrypoint can fan out one container per zip."""
    import os
    vol.reload()
    src = f"{MOUNT}/{SRC_SUBDIR}"
    if not os.path.isdir(src):
        print(f"[extract] ⚠️ '{src}' not found. {MOUNT} contains: {sorted(os.listdir(MOUNT))}")
        return []
    zips = sorted(f for f in os.listdir(src)
                  if f.lower().endswith(".zip") and not f.startswith(".azDownload-"))
    if not zips:
        print(f"[extract] no .zip files under {src}. Contents: {sorted(os.listdir(src))}")
    return zips


@app.function(
    image=image,
    volumes={MOUNT: vol},
    timeout=12 * 3600,
    cpu=2,                        # bsdtar is single-threaded per archive; 2 is plenty
    ephemeral_disk=512 * 1024,    # 512 GiB = Modal's per-container minimum (covers the 185 GiB zip)
)
def extract_one(name: str, delete_zips: bool):
    """Extract ONE zip straight into the volume and commit. Runs in its own container
    (one per zip via .map), so the 4 zips extract concurrently. Disjoint output paths
    => no cross-container conflict on the shared volume."""
    import os
    import subprocess
    import time

    vol.reload()
    zpath = f"{MOUNT}/{SRC_SUBDIR}/{name}"
    zsize = os.path.getsize(zpath)
    print(f"[extract] START {name}  ({_human(zsize)})  -> {MOUNT}/{EXPECTED_ROOT}")
    t0 = time.time()
    # bsdtar first, then unar. subprocess list args -> spaces in names are safe.
    attempts = [
        ["bsdtar", "-x", "-f", zpath, "-C", MOUNT],
        ["unar", "-quiet", "-force-overwrite", "-output-directory", MOUNT, zpath],
    ]
    ok = None
    for cmd in attempts:
        print(f"[extract]   {name}: trying {cmd[0]} ...")
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            ok = cmd[0]
            break
        print(f"[extract]   {name}: {cmd[0]} failed (rc={rc}); trying next ...")
    if ok is None:
        raise RuntimeError(f"extraction of {name!r} failed with all tools; aborting.")
    print(f"[extract]   {name}: done via {ok} in {time.time() - t0:.0f}s")

    if delete_zips:
        os.remove(zpath)
        print(f"[extract]   {name}: removed  (reclaimed {_human(zsize)})")

    print(f"[extract]   {name}: committing volume (persist progress)...")
    vol.commit()
    return name


@app.function(image=image, volumes={MOUNT: vol})
def summarize():
    """Count images under the merged root after all zips have extracted."""
    import os
    vol.reload()
    root = f"{MOUNT}/{EXPECTED_ROOT}"
    print(f"\n[extract] {MOUNT} now contains: {sorted(os.listdir(MOUNT))[:12]}")
    if not os.path.isdir(root):
        print(f"[extract] ⚠️ expected '{root}' not found — check the zips' top-level "
              f"folder name; set remote_data_root to whatever landed under {MOUNT}.")
        return
    per = {}
    for split in ("train", "valid", "test"):
        sp = f"{root}/{split}"
        c = 0
        if os.path.isdir(sp):
            for _dp, _dn, fs in os.walk(sp):
                c += sum(1 for f in fs if f.lower().endswith((".jpg", ".jpeg")))
        per[split] = c
    print(f"[extract] jpg per split under {EXPECTED_ROOT}/: {per}  (total {sum(per.values()):,})")
    print(f"[extract] top-level of {EXPECTED_ROOT}/: {sorted(os.listdir(root))[:12]}")
    print("[extract] done ✅")


@app.local_entrypoint()
def main(delete_zips: bool = False):
    zips = list_zips.remote()
    if not zips:
        return
    print(f"[extract] fanning out {len(zips)} zip(s) in parallel (one container each)")
    # one container per zip, all concurrent; block until every extraction finishes.
    for _ in extract_one.map(zips, [delete_zips] * len(zips)):
        pass
    summarize.remote()
