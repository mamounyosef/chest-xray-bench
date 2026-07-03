r"""
modal_inspect_native.py  —  FAST extraction-verify + resolution profile for the
native-resolution CheXpert tree (/data_native/CheXpert-v1.0/{train,valid}/...).

Speed design (the old version opened ~223k files serially over the network — slow):
  * fan out across many Modal containers with .map() (one shard of patients each),
  * inside each container read image headers with a big THREAD pool (I/O-bound, so
    threads hide the per-file network latency),
  * workers return only tiny AGGREGATES — a {(w,h): count} histogram + mode/filesize
    histograms — never per-image data. That histogram yields EXACT percentiles for
    width / height / aspect / megapixels, so the FULL dataset is cheap and precise.

Runs the FULL dataset by default. Tune everything in the CONFIG block below —
no CLI flags:
    modal run modal_scripts/modal_inspect_native.py
"""

import modal

VOLUME = "chexpert-native-data"
MOUNT = "/data_native"
ROOT_CANDIDATES = ["CheXpert-v1.0", "CheXpert-v1.0-small"]
SEED = 42

# ============================ CONFIG (edit here) ============================
SAMPLE_PATIENTS = 0      # 0 = FULL dataset (default). >0 = random N patient dirs (quick preview)
DECODE_FRAC     = 0.0    # 0 = headers only (fast). e.g. 0.01 = also fully decode 1% (corruption check)
THREADS         = 128    # threads PER container (I/O-bound: high count hides network latency)
SHARD_PATIENTS  = 1500   # patient dirs per shard (-> ~43 shards on the full set; smaller = finer progress)
MAX_CONTAINERS  = 40     # max Modal containers to fan out over (cost cap)
CPU_CORES       = 4.0    # CPU cores reserved per container (parallel header parsing)
# ===========================================================================

app = modal.App("chexpert-inspect-native")
image = modal.Image.debian_slim(python_version="3.11").pip_install("pillow")
vol = modal.Volume.from_name(VOLUME)


# --------------------------------------------------------------------------- #
#  remote: list patient dirs (2 listdirs, fast) so the driver can shard them   #
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={MOUNT: vol}, timeout=30 * 60)
def enumerate_patients():
    import os
    vol.reload()
    root = next((f"{MOUNT}/{c}" for c in ROOT_CANDIDATES if os.path.isdir(f"{MOUNT}/{c}")), None)
    if root is None:
        raise FileNotFoundError(f"none of {ROOT_CANDIDATES} under {MOUNT}: {sorted(os.listdir(MOUNT))}")
    pats = []
    for split in ("train", "valid", "test"):
        d = f"{root}/{split}"
        if os.path.isdir(d):
            pats += [f"{split}/{n}" for n in os.listdir(d)]
    return root, pats


# --------------------------------------------------------------------------- #
#  remote worker: measure one shard of patients with a thread pool             #
# --------------------------------------------------------------------------- #
@app.function(image=image, volumes={MOUNT: vol}, timeout=60 * 60,
              max_containers=MAX_CONTAINERS, cpu=CPU_CORES)
def measure_shard(arg):
    import os
    import random
    from concurrent.futures import ThreadPoolExecutor
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = False   # truncated file -> raises -> counted as corrupt

    root, patients, threads, decode_frac = arg
    exts = (".jpg", ".jpeg", ".png")
    rng = random.Random(SEED)

    def do_patient(rel):
        """Walk one patient dir; return its local aggregates (runs in a thread)."""
        res, modes, fs_hist = {}, {}, {}
        split = rel.split("/", 1)[0]
        n_img = bad = decoded = 0
        for dp, _dn, fs in os.walk(f"{root}/{rel}"):
            for f in fs:
                if not f.lower().endswith(exts):
                    continue
                p = os.path.join(dp, f)
                n_img += 1
                try:
                    with Image.open(p) as im:
                        wh = im.size          # (w, h) — header only, no pixel decode
                        m = im.mode
                        if decode_frac and rng.random() < decode_frac:
                            im.load()         # force full decode (corruption check)
                            decoded += 1
                    res[wh] = res.get(wh, 0) + 1
                    modes[m] = modes.get(m, 0) + 1
                    kib = os.path.getsize(p) // 1024
                    fs_hist[kib] = fs_hist.get(kib, 0) + 1
                except Exception:
                    bad += 1
        return split, n_img, bad, decoded, res, modes, fs_hist

    # merge all patients in this shard
    res, modes, fs_hist, per_split = {}, {}, {}, {}
    total = bad = decoded = 0
    with ThreadPoolExecutor(max_workers=threads) as ex:
        for split, n_img, b, dec, r, mo, fh in ex.map(do_patient, patients):
            total += n_img; bad += b; decoded += dec
            per_split[split] = per_split.get(split, 0) + n_img
            for k, v in r.items():
                res[k] = res.get(k, 0) + v
            for k, v in mo.items():
                modes[k] = modes.get(k, 0) + v
            for k, v in fh.items():
                fs_hist[k] = fs_hist.get(k, 0) + v
    return {"total": total, "bad": bad, "decoded": decoded, "per_split": per_split,
            "res": res, "modes": modes, "fs_hist": fs_hist}


# --------------------------------------------------------------------------- #
#  driver (local): shard, fan out, merge, report exact stats                   #
# --------------------------------------------------------------------------- #
def _wpcts(pairs, ps):
    """Exact percentiles from (value, weight) pairs. Returns {p: value}."""
    items = sorted(pairs)
    total = sum(w for _v, w in items)
    out, cum, ti = {}, 0, 0
    sps = sorted(ps)
    for v, w in items:
        cum += w
        while ti < len(sps) and cum >= sps[ti] / 100 * total:
            out[sps[ti]] = v
            ti += 1
    while ti < len(sps):
        out[sps[ti]] = items[-1][0] if items else 0
        ti += 1
    return out


def _line(name, pairs):
    ps = [0, 5, 25, 50, 75, 95, 100]
    q = _wpcts(pairs, ps)
    total = sum(w for _v, w in pairs) or 1
    mean = sum(v * w for v, w in pairs) / total
    fmt = "{:.3f}" if any(isinstance(v, float) for v, _ in pairs) else "{:.0f}"
    cells = "  ".join(f"p{p}=" + fmt.format(q[p]) for p in ps)
    return f"  {name:<11}: {cells}  mean={mean:.3f}"


def _bar(frac, width=24):
    n = int(frac * width)
    return "[" + "#" * n + "-" * (width - n) + "]"


@app.local_entrypoint()
def main():
    import random
    import time
    root, pats = enumerate_patients.remote()
    print(f"[inspect] root={root}   patient dirs={len(pats):,}")
    if SAMPLE_PATIENTS and SAMPLE_PATIENTS < len(pats):
        pats = random.Random(SEED).sample(pats, SAMPLE_PATIENTS)
        print(f"[inspect] sampling {len(pats):,} patient dirs (SAMPLE_PATIENTS)")
    else:
        print("[inspect] FULL dataset")

    shards = [(root, pats[i:i + SHARD_PATIENTS], THREADS, DECODE_FRAC)
              for i in range(0, len(pats), SHARD_PATIENTS)]
    print(f"[inspect] {len(shards)} shards x ~{SHARD_PATIENTS} patients, {THREADS} threads/container, "
          f"{CPU_CORES} CPU/container, map cap={MAX_CONTAINERS}  (decode_frac={DECODE_FRAC})")

    res, modes, fs_hist, per_split = {}, {}, {}, {}
    total = bad = decoded = 0
    done = 0
    n = len(shards)
    t0 = time.time()
    for r in measure_shard.map(shards):
        total += r["total"]; bad += r["bad"]; decoded += r["decoded"]
        for d, src in ((per_split, r["per_split"]), (res, r["res"]),
                       (modes, r["modes"]), (fs_hist, r["fs_hist"])):
            for k, v in src.items():
                d[k] = d.get(k, 0) + v
        done += 1
        frac = done / n
        el = time.time() - t0
        eta = el / frac - el if frac > 0 else 0
        print(f"[inspect] {_bar(frac)} {done}/{n} ({100*frac:4.1f}%)  "
              f"images={total:,}  {total/max(el,1e-9):,.0f} img/s  "
              f"elapsed={el:.0f}s  ETA~{eta:.0f}s")

    # derive weighted value lists from the (w,h) histogram -> EXACT percentiles
    w_pairs = {}
    h_pairs = {}
    ar_pairs = []
    mp_pairs = []
    for (w, h), c in res.items():
        w_pairs[w] = w_pairs.get(w, 0) + c
        h_pairs[h] = h_pairs.get(h, 0) + c
        ar_pairs.append((round(w / h, 3), c))
        mp_pairs.append((round(w * h / 1e6, 3), c))

    print("\n" + "=" * 82)
    print(f"NATIVE CheXpert RESOLUTION PROFILE   images={total:,}"
          + (f"   unreadable={bad:,}" if bad else "")
          + (f"   decoded-checked={decoded:,}" if decoded else ""))
    print("=" * 82)
    print(f"  per split   : {per_split}")
    print(_line("width px", list(w_pairs.items())))
    print(_line("height px", list(h_pairs.items())))
    print(_line("aspect w/h", ar_pairs))
    print(_line("megapixels", mp_pairs))
    print(_line("file KiB", list(fs_hist.items())))
    print(f"  colour modes: {modes}")
    print(f"  unique (w x h): {len(res):,}")
    print("  top 15 resolutions (w x h : count, %):")
    for (w, h), c in sorted(res.items(), key=lambda kv: kv[1], reverse=True)[:15]:
        print(f"     {w:>5} x {h:<5} : {c:>8,}  ({100*c/max(total,1):5.2f}%)  aspect {w/h:.3f}")
    print("=" * 82)
    if bad:
        print(f"[inspect] ⚠️ {bad:,} unreadable/corrupt image(s).")
    portrait = sum(c for (w, h), c in res.items() if h > w)
    print(f"[inspect] portrait share (h>w): {100*portrait/max(total,1):.1f}%  "
          f"-> use the percentiles + top resolutions to pick the native-run geometry.")
