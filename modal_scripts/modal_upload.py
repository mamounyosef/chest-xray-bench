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

# If True, delete the uploaded archive from the volume after extracting it.
# Defaults to False = KEEP the archive on the volume after extraction.
DELETE_ARCHIVE_AFTER_EXTRACT = False

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
              cpu=12.0, memory=8192, timeout=6 * 3600)
def extract(archive: str, expected: str = "CheXpert-v1.0-small", mount: str = "/data"):
    import collections
    import os
    import subprocess
    import tarfile
    import time
    import zipfile
    from concurrent.futures import ThreadPoolExecutor

    def _fmt(sec):
        sec = int(max(0, sec))
        h, r = divmod(sec, 3600); m, s = divmod(r, 60)
        return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"

    def _parallel_write(entries, dest_root, total=None, total_bytes=None,
                        workers=512, window=2048):
        """Write files concurrently to hide per-file volume latency.

        `entries` yields (relpath, data_bytes) read sequentially from the single
        archive stream (main thread); the actual disk writes — the latency-bound
        part on a network volume — run on a thread pool. A bounded in-flight
        window caps memory (~window * avg_file). Returns the file count written.

        Prints a detailed line every ~5s: files done + %, GB done + %, current
        files/s and MB/s, average files/s, elapsed, and ETA (when total known).
        """
        root = os.path.realpath(dest_root)

        def _w(item):
            rel, data = item
            full = os.path.realpath(os.path.join(dest_root, rel))
            if full != root and not full.startswith(root + os.sep):
                raise ValueError(f"unsafe path escapes {dest_root!r}: {rel!r}")
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as fh:
                fh.write(data)
            return len(data)

        print(f"[extract] writing with {workers} parallel writers, window {window}"
              + (f" | target {total} files" if total else "")
              + (f", {total_bytes / 1e9:.2f} GB" if total_bytes else ""), flush=True)
        t0 = last = time.time()
        n = done_bytes = last_n = last_bytes = 0
        it = iter(entries)
        inflight = collections.deque()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for _ in range(window):
                item = next(it, None)
                if item is None:
                    break
                inflight.append(ex.submit(_w, item))
            while inflight:
                done_bytes += inflight.popleft().result()
                item = next(it, None)
                if item is not None:
                    inflight.append(ex.submit(_w, item))
                n += 1
                now = time.time()
                if now - last >= 5.0 or (total and n == total):
                    el, dt = now - t0, now - last
                    avg = n / max(1e-6, el)
                    inst = (n - last_n) / max(1e-6, dt)
                    mbs = (done_bytes - last_bytes) / 1e6 / max(1e-6, dt)
                    files_part = (f"{n}/{total} ({100 * n / total:.1f}%)"
                                  if total else f"{n}")
                    gb_part = (f"{done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB "
                               f"({100 * done_bytes / max(1, total_bytes):.1f}%)"
                               if total_bytes else f"{done_bytes / 1e9:.2f} GB")
                    eta = f" | ETA {_fmt((total - n) / max(1e-6, avg))}" if total else ""
                    print(f"[extract]   {files_part} | {gb_part} | "
                          f"now {inst:.0f} files/s, {mbs:.0f} MB/s | "
                          f"avg {avg:.0f} files/s | elapsed {_fmt(el)}{eta}", flush=True)
                    last, last_n, last_bytes = now, n, done_bytes
        return n

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
        # Pre-scan (headers only, no data read) to learn file count + total bytes
        # so the write pass can show a % and an ETA. This is a single streaming
        # pass over the headers (~seconds even for 200k+ members).
        print(f"[extract] tar: pre-scanning headers for count + size ...", flush=True)
        ts = last = time.time()
        n_total = 0
        bytes_total = 0
        with tarfile.open(path, "r|") as tf:
            for m in tf:
                if m.isreg():
                    n_total += 1
                    bytes_total += m.size
                now = time.time()
                if now - last >= 3.0:
                    print(f"[extract]   ...scanned {n_total} files, "
                          f"{bytes_total / 1e9:.2f} GB", flush=True)
                    last = now
        print(f"[extract] tar: {n_total} files, {bytes_total / 1e9:.2f} GB "
              f"(scan {_fmt(time.time() - ts)})", flush=True)

        # Second pass: read the stream sequentially, write files in parallel.
        def _tar_entries():
            with tarfile.open(path, "r|") as tf:
                for m in tf:
                    if m.isreg():
                        yield m.name, tf.extractfile(m).read()
                    elif m.isdir():
                        os.makedirs(os.path.join(mount, m.name), exist_ok=True)
        print(f"[extract] tar: reading stream + writing files in parallel ...", flush=True)
        n = _parallel_write(_tar_entries(), mount, total=n_total, total_bytes=bytes_total)
        kind = "tar"
    elif zipfile.is_zipfile(path):
        # ZipFile isn't thread-safe for concurrent reads on one handle, so read
        # sequentially in the main thread and write in parallel.
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            total = sum(1 for i in infos if not i.is_dir())
            bytes_total = sum(i.file_size for i in infos if not i.is_dir())
            for i in infos:
                if i.is_dir():
                    os.makedirs(os.path.join(mount, i.filename), exist_ok=True)
            def _zip_entries():
                for i in infos:
                    if not i.is_dir():
                        yield i.filename, zf.read(i)
            print(f"[extract] zip: {total} files, {bytes_total / 1e9:.2f} GB; "
                  f"reading + writing in parallel ...", flush=True)
            n = _parallel_write(_zip_entries(), mount, total=total, total_bytes=bytes_total)
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

    if DELETE_ARCHIVE_AFTER_EXTRACT:
        os.remove(path)                                # drop the archive, keep the files
        print("[extract] removed the archive; committing volume (indexing files)...")
    else:
        print(f"[extract] keeping archive {path} (DELETE_ARCHIVE_AFTER_EXTRACT=False); "
              f"committing volume (indexing files)...")
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
