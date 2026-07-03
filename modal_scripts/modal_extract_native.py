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
It commits the volume after EACH zip so progress survives a restart.

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


@app.function(
    image=image,
    volumes={MOUNT: vol},
    timeout=12 * 3600,
    ephemeral_disk=1024 * 1024,   # 1 TiB scratch (bsdtar buffering headroom)
)
def extract_all(delete_zips: bool):
    import os
    import subprocess
    import time

    vol.reload()
    src = f"{MOUNT}/{SRC_SUBDIR}"
    if not os.path.isdir(src):
        print(f"[extract] ⚠️ '{src}' not found. {MOUNT} contains: {sorted(os.listdir(MOUNT))}")
        return

    # clean .zip archives only (skip azcopy's .azDownload-* partials, if any linger)
    zips = sorted(f for f in os.listdir(src)
                  if f.lower().endswith(".zip") and not f.startswith(".azDownload-"))
    if not zips:
        print(f"[extract] no .zip files under {src}. Contents: {sorted(os.listdir(src))}")
        return
    print(f"[extract] target = {MOUNT}/{EXPECTED_ROOT}  (merging {len(zips)} zip(s))")

    for i, name in enumerate(zips, 1):
        zpath = f"{src}/{name}"
        zsize = os.path.getsize(zpath)
        print(f"\n[extract] ({i}/{len(zips)}) {name}  ({_human(zsize)})")
        t0 = time.time()
        # bsdtar first, then unar. subprocess list args -> spaces in names are safe.
        attempts = [
            ["bsdtar", "-x", "-f", zpath, "-C", MOUNT],
            ["unar", "-quiet", "-force-overwrite", "-output-directory", MOUNT, zpath],
        ]
        ok = None
        for cmd in attempts:
            print(f"[extract]   trying {cmd[0]} ...")
            rc = subprocess.run(cmd).returncode
            if rc == 0:
                ok = cmd[0]
                break
            print(f"[extract]   {cmd[0]} failed (rc={rc}); trying next ...")
        if ok is None:
            raise RuntimeError(f"extraction of {name!r} failed with all tools; aborting.")
        print(f"[extract]   done via {ok} in {time.time() - t0:.0f}s")

        if delete_zips:
            os.remove(zpath)
            print(f"[extract]   removed {name}  (reclaimed {_human(zsize)})")

        print("[extract]   committing volume (persist progress)...")
        vol.commit()

    # summary: count images under the merged root
    root = f"{MOUNT}/{EXPECTED_ROOT}"
    print(f"\n[extract] {MOUNT} now contains: {sorted(os.listdir(MOUNT))[:12]}")
    if not os.path.isdir(root):
        print(f"[extract] ⚠️ expected '{root}' not found — check the zips' top-level "
              f"folder name; set remote_data_root to whatever landed under {MOUNT}.")
        vol.commit()
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
    vol.commit()
    print("[extract] committed — done ✅")


@app.local_entrypoint()
def main(delete_zips: bool = False):
    extract_all.remote(delete_zips)
