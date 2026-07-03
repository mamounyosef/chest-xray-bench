r"""
modal_consolidate_native.py  —  merge the 4 extracted CheXpert batch folders into
the standard tree the CSVs expect:  /data_native/CheXpert-v1.0/{train,valid}/...

After extraction the data is spread across (folder names have spaces + parens):
    CheXpert-v1.0 batch 1 (validate & csv)/valid/patientNNNNN/...   + train.csv, valid.csv
    CheXpert-v1.0 batch 2 (train 1)/patientNNNNN/...                (train, no wrapper)
    CheXpert-v1.0 batch 3 (train 2)/patientNNNNN/...                (train)
    CheXpert-v1.0 batch 4 (train 3)/patientNNNNN/...                (train)
but the CSV Path column is  'CheXpert-v1.0/train/patientNNNNN/...'  and
'CheXpert-v1.0/valid/patientNNNNN/...'. This script moves everything into that
exact layout so Path matches disk with NO CSV edits.

Everything is on ONE volume, so each move is an instant os.rename (metadata only —
no 438 GB re-copy). Moves don't lose data; only empty source dirs are removed.

  -> after this, set remote_data_root: /data_native/CheXpert-v1.0

Run (preview only, no changes):
    modal run modal_scripts/modal_consolidate_native.py --dry-run
Run (apply the moves):
    modal run modal_scripts/modal_consolidate_native.py
"""

import modal

VOLUME = "chexpert-native-data"
MOUNT = "/data_native"
TARGET = "CheXpert-v1.0"                       # final root folder (matches CSV Path prefix)
VALID_SRC = "CheXpert-v1.0 batch 1 (validate & csv)"
TRAIN_SRCS = ["CheXpert-v1.0 batch 2 (train 1)",
              "CheXpert-v1.0 batch 3 (train 2)",
              "CheXpert-v1.0 batch 4 (train 3)"]
CSVS = ["train.csv", "valid.csv"]
LEFTOVER_DIR = "chexpertchestxrays-u20210408"  # empty azcopy container dir to clean up

app = modal.App("chexpert-consolidate-native")
image = modal.Image.debian_slim(python_version="3.11")
vol = modal.Volume.from_name(VOLUME)


@app.function(image=image, volumes={MOUNT: vol}, timeout=6 * 3600)
def consolidate(dry_run: bool):
    import os

    vol.reload()
    tag = "[dry-run] would" if dry_run else "[apply]"

    def move(src, dst):
        """os.rename src->dst (same volume => instant). Skips if dst exists."""
        if os.path.exists(dst):
            print(f"  ⚠️ target exists, SKIP: {dst}")
            return 0
        if not dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(src, dst)
        return 1

    train_dir = f"{MOUNT}/{TARGET}/train"
    valid_dir = f"{MOUNT}/{TARGET}/valid"
    if not dry_run:
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(valid_dir, exist_ok=True)

    # ---- train: move patient dirs from the 3 train batches into TARGET/train ----
    n_train = 0
    for sub in TRAIN_SRCS:
        src_root = f"{MOUNT}/{sub}"
        if not os.path.isdir(src_root):
            print(f"[train] source not found (already moved?): {sub}")
            continue
        entries = sorted(os.listdir(src_root))
        pats = [e for e in entries if e.startswith("patient")]
        other = [e for e in entries if not e.startswith("patient")]
        if other:
            print(f"[train] ⚠️ non-patient entries in {sub}: {other}")
        print(f"[train] {tag} move {len(pats):,} patient dirs from '{sub}' -> {TARGET}/train/")
        for e in pats:
            n_train += move(f"{src_root}/{e}", f"{train_dir}/{e}")

    # ---- valid: move patient dirs from batch 1's valid/ into TARGET/valid ----
    n_valid = 0
    vsrc = f"{MOUNT}/{VALID_SRC}/valid"
    if os.path.isdir(vsrc):
        pats = sorted(e for e in os.listdir(vsrc) if e.startswith("patient"))
        print(f"[valid] {tag} move {len(pats):,} patient dirs -> {TARGET}/valid/")
        for e in pats:
            n_valid += move(f"{vsrc}/{e}", f"{valid_dir}/{e}")
    else:
        print(f"[valid] source not found (already moved?): {vsrc}")

    # ---- CSVs: move to TARGET/ (Path column already references CheXpert-v1.0/...) ----
    for c in CSVS:
        csrc = f"{MOUNT}/{VALID_SRC}/{c}"
        if os.path.isfile(csrc):
            print(f"[csv] {tag} move {c} -> {TARGET}/{c}")
            move(csrc, f"{MOUNT}/{TARGET}/{c}")

    # ---- clean up emptied source dirs + leftover azcopy container dir ----
    if not dry_run:
        for sub in TRAIN_SRCS + [f"{VALID_SRC}/valid", VALID_SRC, LEFTOVER_DIR]:
            p = f"{MOUNT}/{sub}"
            try:
                if os.path.isdir(p) and not os.listdir(p):
                    os.rmdir(p)
                    print(f"[cleanup] removed empty dir: {sub}")
                elif os.path.isdir(p):
                    print(f"[cleanup] NOT empty, left in place: {sub} -> {os.listdir(p)[:5]}")
            except Exception as e:
                print(f"[cleanup] could not remove {sub}: {e!r}")

    print(f"\n{tag} moved train={n_train:,}  valid={n_valid:,} patient dirs")
    if not dry_run:
        print("[apply] committing volume...")
        vol.commit()
        print(f"[apply] done ✅  root is now {MOUNT}/{TARGET}/{{train,valid}}/")
        print(f"[apply] {MOUNT} now: {sorted(os.listdir(MOUNT))}")
    else:
        print("[dry-run] no changes made. Re-run without --dry-run to apply.")


@app.local_entrypoint()
def main(dry_run: bool = False):
    consolidate.remote(dry_run)
