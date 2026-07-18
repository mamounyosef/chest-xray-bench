r"""
modal_download.py  —  the REVERSE of modal_upload.py.

modal_upload.py takes a local archive, uploads it, and extracts ~200k tiny JPEGs
onto a Modal volume. This script does the opposite: the images already live
EXTRACTED on the volume, and you want them back on your PC as ONE file. Pulling
200k tiny files one-by-one with `modal volume get <dir>` is slow and latency-bound,
so instead we:

  1. (Modal, server-side) tar the dataset folder ON the volume into ONE archive
     written back to the volume root  (Store / no compression — JPEGs don't shrink).
  2. (local) download that single archive with `modal volume get` (one big
     sequential transfer) to the path you choose.
  3. (Modal) delete the archive from the volume to reclaim space
     (skip with --keep-archive).

Both steps run from ONE command — the tar happens remotely, the download happens
locally (the local entrypoint shells out to `modal volume get`).

  CheXpert (default):
     modal run modal_download.py --dest "D:/chexpert.tar"
  ChestX-ray14 (other volume):
     modal run modal_download.py --source images --mount /data_cxr14 \
         --archive chestxray14.tar --dest "D:/chestxray14.tar"

After download, extract locally with any tar tool (7-Zip, `tar -xf`, bsdtar).

TWO volumes are mounted so this one script handles both datasets:
    chexpert-data     -> /data       (default;  CheXpert)
    chestxray14-data  -> /data_cxr14  (pass --mount /data_cxr14;  ChestX-ray14)
"""

import modal

# (volume name, mount path) for each dataset volume — mirrors modal_upload.py.
DATA_VOLUME = "chexpert-data"
CXR14_VOLUME = "chestxray14-data"

app = modal.App("chexpert-download")
image = modal.Image.debian_slim(python_version="3.11")
vol = modal.Volume.from_name(DATA_VOLUME, create_if_missing=True)
vol_cxr14 = modal.Volume.from_name(CXR14_VOLUME, create_if_missing=True)
_VOL_BY_MOUNT = {"/data": vol, "/data_cxr14": vol_cxr14}
_VOLNAME_BY_MOUNT = {"/data": DATA_VOLUME, "/data_cxr14": CXR14_VOLUME}


@app.function(image=image, volumes={"/data": vol, "/data_cxr14": vol_cxr14},
              cpu=12.0, memory=8192, timeout=6 * 3600)
def make_archive(source: str = "CheXpert-v1.0-small", archive: str = "chexpert.tar",
                 mount: str = "/data"):
    """Tar `<mount>/<source>` into `<mount>/<archive>` on the volume, commit, and
    return (archive_name, size_bytes, n_files). Store mode (no compression)."""
    import collections
    import io
    import os
    import tarfile
    import time
    from concurrent.futures import ThreadPoolExecutor

    target_vol = _VOL_BY_MOUNT.get(mount)
    if target_vol is None:
        raise ValueError(f"unknown --mount {mount!r}; expected one of {list(_VOL_BY_MOUNT)}")

    ts_wall = time.time()      # wall-clock start for the whole server-side job
    print(f"[archive] reloading volume metadata for mount {mount} ...", flush=True)
    target_vol.reload()
    src = f"{mount}/{source}"
    out = f"{mount}/{archive}"
    if not os.path.isdir(src):
        avail = sorted(os.listdir(mount))[:12]
        raise FileNotFoundError(
            f"source folder {src!r} not found on the volume mounted at {mount}. "
            f"Contents: {avail}. Pass --source with the correct top-level folder.")

    def _fmt(sec):
        sec = int(max(0, sec))
        h, r = divmod(sec, 3600); m, s = divmod(r, 60)
        return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"

    # ------------------------------------------------------------------ scan --
    # pre-scan so the tar pass can show a percentage + ETA. On a network-backed
    # Modal volume this walk is itself latency-bound (one stat() per inode over
    # ~478k files), so it can take MINUTES — hence throttled live progress here
    # instead of a single silent print at the end.
    print(f"[scan] walking {src} to count files + bytes "
          f"(this can take a while on a network volume) ...", flush=True)
    ts = last = time.time()
    total_files = total_bytes = 0
    n_dirs = 0
    entries = []       # (full_path, arcname) collected once, reused by the tar pass
    for root, _dirs, files in os.walk(src):
        n_dirs += 1
        for f in files:
            total_files += 1
            full = os.path.join(root, f)
            entries.append((full, os.path.relpath(full, mount)))
            try:
                total_bytes += os.path.getsize(full)
            except OSError:
                pass
            now = time.time()
            if now - last >= 3.0:      # heartbeat every ~3s so it never looks hung
                el = now - ts
                fps = total_files / max(1e-6, el)
                print(f"[scan]   ...{total_files} files, {n_dirs} dirs, "
                      f"{total_bytes / 1e9:.2f} GB so far | {fps:.0f} files/s | "
                      f"elapsed {_fmt(el)}", flush=True)
                last = now
    scan_secs = time.time() - ts
    print(f"[scan] DONE: {total_files} files in {n_dirs} dirs, "
          f"{total_bytes / 1e9:.2f} GB, in {_fmt(scan_secs)} "
          f"({total_files / max(1e-6, scan_secs):.0f} files/s)", flush=True)

    # ------------------------------------------------------------------ tar ---
    # THE speed fix. Serial tar was ~9 files/s: every tiny JPEG costs one full
    # network round-trip to the volume, and tarfile reads them one at a time. We
    # hide that latency by reading many files CONCURRENTLY with a thread pool
    # (socket reads release the GIL), while still writing the tar serially in the
    # main thread. A bounded in-flight window caps memory (~WINDOW * avg_file).
    # Latency-bound, not CPU-bound: threads mostly wait on volume round-trips, so
    # we oversubscribe far past the core count. cpu=8 on the Function just raises
    # the network/bandwidth ceiling these readers can fill.
    WORKERS = 512      # concurrent reads
    WINDOW = 2048      # max files buffered in RAM ahead of the writer

    def _read(entry):
        full, arcname = entry
        with open(full, "rb") as fh:
            return arcname, fh.read()

    print(f"[tar] packing {src}  ->  {out}  (Store / no compression) | "
          f"{WORKERS} parallel readers, window {WINDOW}", flush=True)
    print(f"[tar] progress prints every ~5s: files done, GB done, throughput, "
          f"files/s, elapsed, ETA.", flush=True)
    t0 = last = time.time()
    n = done_bytes = 0
    last_n = last_bytes = 0        # for instantaneous (not just average) rates
    it = iter(entries)
    inflight = collections.deque()
    with tarfile.open(out, "w") as tf, ThreadPoolExecutor(max_workers=WORKERS) as ex:
        # prime the pipeline
        for _ in range(WINDOW):
            e = next(it, None)
            if e is None:
                break
            inflight.append(ex.submit(_read, e))
        # drain: write oldest result, top the window back up
        while inflight:
            arcname, data = inflight.popleft().result()
            e = next(it, None)
            if e is not None:
                inflight.append(ex.submit(_read, e))
            ti = tarfile.TarInfo(name=arcname)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
            n += 1
            done_bytes += len(data)
            now = time.time()
            if now - last >= 5.0 or n == total_files:   # throttle to ~every 5s
                el = now - t0
                dt = now - last
                avg_fps = n / max(1e-6, el)
                inst_fps = (n - last_n) / max(1e-6, dt)          # right-now files/s
                inst_mbs = (done_bytes - last_bytes) / 1e6 / max(1e-6, dt)  # right-now MB/s
                eta = (total_files - n) / max(1e-6, avg_fps)
                print(f"[tar]   {n}/{total_files} files "
                      f"({100 * n / max(1, total_files):.1f}%) | "
                      f"{done_bytes / 1e9:.2f}/{total_bytes / 1e9:.2f} GB | "
                      f"now {inst_fps:.0f} files/s, {inst_mbs:.0f} MB/s | "
                      f"avg {avg_fps:.0f} files/s | "
                      f"elapsed {_fmt(el)} | ETA {_fmt(eta)}", flush=True)
                last, last_n, last_bytes = now, n, done_bytes
    size = os.path.getsize(out)
    tar_secs = time.time() - t0
    print(f"[tar] DONE wrote {archive}: {n} files, {size / 1e9:.2f} GB "
          f"in {_fmt(tar_secs)} ({size / 1e6 / max(1e-6, tar_secs):.0f} MB/s, "
          f"{n / max(1e-6, tar_secs):.0f} files/s)", flush=True)

    # ---------------------------------------------------------------- commit --
    print("[commit] committing volume so the new archive is indexed + "
          "downloadable (can take a while for a multi-GB file)...", flush=True)
    t1 = time.time()
    target_vol.commit()
    print(f"[commit] DONE in {_fmt(time.time() - t1)} ✅", flush=True)
    print(f"[archive] TOTAL server-side time: {_fmt(time.time() - ts_wall)} "
          f"(scan {_fmt(scan_secs)} + tar {_fmt(tar_secs)} + commit)", flush=True)
    return archive, size, n


@app.function(image=image, volumes={"/data": vol, "/data_cxr14": vol_cxr14},
              timeout=3600)
def remove_archive(archive: str, mount: str = "/data"):
    """Delete `<mount>/<archive>` from the volume and commit (reclaim space)."""
    import os

    target_vol = _VOL_BY_MOUNT.get(mount)
    target_vol.reload()
    path = f"{mount}/{archive}"
    if os.path.exists(path):
        os.remove(path)
        target_vol.commit()
        print(f"[cleanup] removed {path} from the volume ✅", flush=True)
    else:
        print(f"[cleanup] {path} already gone", flush=True)


@app.local_entrypoint()
def main(source: str = "CheXpert-v1.0-small", archive: str = "chexpert.tar",
         mount: str = "/data", dest: str = None, keep_archive: bool = False):
    """
    source       : folder ON the volume to compress (top-level dataset dir).
    archive      : name of the single archive to create (and download).
    mount        : which volume — /data (CheXpert) or /data_cxr14 (ChestX-ray14).
    dest         : LOCAL path to save the archive (default: ./<archive>).
    keep_archive : leave the archive on the volume instead of deleting it after.
    """
    import subprocess
    import time

    # 1) build the archive server-side on the volume.
    name, size, n = make_archive.remote(source, archive, mount)

    # 2) download that single file locally via the modal CLI (runs on THIS machine).
    volname = _VOLNAME_BY_MOUNT[mount]
    local_dest = dest or f"./{name}"
    print(f"\n[download] pulling {name} ({size / 1e9:.2f} GB, {n} files) "
          f"from volume {volname!r} -> {local_dest}", flush=True)
    t0 = time.time()
    subprocess.run(["modal", "volume", "get", "--force", volname, name, local_dest],
                   check=True)
    print(f"[download] done in {time.time() - t0:.0f}s -> {local_dest}", flush=True)

    # 3) reclaim the space on the volume unless asked to keep it.
    if keep_archive:
        print(f"[cleanup] --keep-archive set; leaving {name} on the volume.")
    else:
        remove_archive.remote(name, mount)

    print(f"\n✅ archive saved to {local_dest}. Extract it locally with any tar tool "
          f"(7-Zip, `tar -xf {name}`, or bsdtar).")
