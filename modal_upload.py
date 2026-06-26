r"""
modal_upload.py  —  fast one-shot dataset ingestion for the Modal data volumes.

Uploading ~200k tiny JPEGs file-by-file with `modal volume put <dir>` is slow and
can fail in post-processing. The fast, official pattern is to upload ONE archive
and extract it server-side. Workflow:

  1. (local) make ONE archive of the dataset folder. ZIP/TAR are handled
     natively; RAR is handled via bsdtar/unar (installed in the image).
     Prefer ZIP/TAR with Store/no compression (JPEGs don't shrink).
       e.g.  D:\chexpert.zip  or  D:\CheXpert-v1.0-small.rar
  2. (local) upload the single archive (one big sequential transfer):
       modal volume put chexpert-data "D:/chexpert.zip" /chexpert.zip
  3. extract it onto the volume, server-side, and commit  (pass the EXACT name
     you uploaded it as):
       modal run modal_upload.py --archive chexpert.zip
       modal run modal_upload.py --archive chexpert.rar

After this, /CheXpert-v1.0-small/... exists on the volume (matches remote_data_root).
Upload the small CSVs normally with `modal volume put` (they are few files).

TWO volumes are mounted so this one script handles both datasets (each Modal
volume has a 500k-inode cap, so ChestX-ray14's 112k images get their OWN volume):
    chexpert-data     -> /data       (default;  CheXpert)
    chestxray14-data  -> /data_cxr14  (pass --mount /data_cxr14;  ChestX-ray14)
The archive must already be uploaded to the SAME volume you extract into.
"""

import modal

# (volume name, mount path) for each dataset volume. `extract` picks one by --mount.
DATA_VOLUME = "chexpert-data"
CXR14_VOLUME = "chestxray14-data"

app = modal.App("chexpert-upload")
# RAR is extracted via bsdtar (libarchive, good RAR5 support) with unar as a
# fallback; both are free and in debian main. zip/tar use Python directly.
image = modal.Image.debian_slim(python_version="3.11").apt_install("libarchive-tools", "unar")
vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
vol_cxr14 = modal.Volume.from_name(CXR14_VOLUME, create_if_missing=True)
_VOL_BY_MOUNT = {"/data": vol, "/data_cxr14": vol_cxr14}


@app.function(image=image, volumes={"/data": vol, "/data_cxr14": vol_cxr14},
              timeout=6 * 3600)
def extract(archive: str, expected: str = "CheXpert-v1.0-small", mount: str = "/data"):
    import os
    import subprocess
    import tarfile
    import time
    import zipfile

    target_vol = _VOL_BY_MOUNT.get(mount)
    if target_vol is None:
        raise ValueError(f"unknown --mount {mount!r}; expected one of {list(_VOL_BY_MOUNT)}")

    target_vol.reload()
    path = f"{mount}/{archive}"
    print(f"[extract] opening {path}  (volume mounted at {mount})")
    if not os.path.exists(path):
        avail = os.listdir(mount)
        raise FileNotFoundError(
            f"{path} not found on the volume mounted at {mount}. Files there: {avail}. "
            f"Pass the exact uploaded name, e.g. --archive {avail[0] if avail else '<name>'}")

    t0 = time.time()
    if tarfile.is_tarfile(path):
        with tarfile.open(path) as tf:
            try:
                tf.extractall(mount, filter="data")    # py3.12+ safe filter
            except TypeError:
                tf.extractall(mount)
            n = len(tf.getnames())
        kind = "tar"
    elif zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(mount)                        # zip64 (>4GB) supported
            n = len(zf.namelist())
        kind = "zip"
    else:
        # RAR (or anything else): try bsdtar (libarchive, solid RAR5 support),
        # then unar, then unrar. Accept the first that exits 0 AND produces the
        # expected folder.
        exp_dir = f"{mount}/{expected}"
        attempts = [
            ["bsdtar", "-x", "-f", path, "-C", mount],
            ["unar", "-quiet", "-force-overwrite", "-output-directory", mount, path],
        ]
        kind = None
        for cmd in attempts:
            print(f"[extract] not zip/tar; trying {cmd[0]} ...")
            rc = subprocess.run(cmd).returncode
            if rc == 0 and os.path.isdir(exp_dir):
                kind = cmd[0]
                break
            print(f"[extract] {cmd[0]} failed (rc={rc}); trying next ...")
        if kind is None:
            raise RuntimeError(
                "RAR extraction failed with all available tools. Re-create the "
                "archive as ZIP (Store, no compression) or TAR and re-upload — "
                "those extract reliably on Linux.")
        n = -1
    print(f"[extract] extracted ({kind}) in {time.time() - t0:.0f}s"
          + (f", {n} entries" if n >= 0 else ""))

    # sanity: confirm the dataset landed where the config expects it
    print(f"[extract] {mount} now contains: {sorted(os.listdir(mount))[:12]}")
    exp_dir = f"{mount}/{expected}"
    if not os.path.isdir(exp_dir):
        print(f"[extract] ⚠️  expected '{exp_dir}' not found — check the archive's "
              f"top-level folder name (remote_data_root must contain /{expected}).")

    os.remove(path)                                    # drop the archive, keep the files
    print("[extract] removed the archive; committing volume (indexing files)...")
    t1 = time.time()
    target_vol.commit()
    print(f"[extract] committed in {time.time() - t1:.0f}s — done ✅")


@app.local_entrypoint()
def main(archive: str = "chexpert.zip", expected: str = "CheXpert-v1.0-small",
         mount: str = "/data"):
    # `expected` = the archive's top-level folder to verify after extract.
    # `mount`    = which volume to extract into (/data = chexpert-data default;
    #              /data_cxr14 = chestxray14-data).
    #   CheXpert     :  modal run modal_upload.py --archive chexpert.rar
    #   ChestX-ray14 :  modal run modal_upload.py --archive chestxray14.rar \
    #                       --expected images --mount /data_cxr14
    extract.remote(archive, expected, mount)
