r"""
modal_download_medmae.py  —  download the Medical-MAE ViT-B/16 checkpoints
(lambert-x/medical_mae) straight onto the Modal data volumes, so training reads them
locally on Modal and they never touch your PC.

Downloads 3 files by Google-Drive file id into  <mount>/pretrained/ :
    vit-b_CXR_0.5M_mae.pth            <- Run 1  (raw MAE self-supervised backbone)
    vit-b_CXR_0.5M_mae_nih14.pth      <- Run 2  (NIH ChestX-ray14 fine-tuned, mAUC 83.0)
    vit-b_CXR_0.5M_mae_chexpert.pth   <- Run 3  (CheXpert fine-tuned, mAUC 89.3)

TARGETS BOTH volumes when both exist:
    chexpert-data        /data/pretrained          (384x320 runs)
    chexpert-native-data /data_native/pretrained   (768x640 staged runs — the ViT
                                                     stage runs mount ONLY the native
                                                     volume, so the checkpoint must
                                                     live here too)
The native volume is included ONLY if it already exists (a run created it); it is
never created just for the checkpoints. Each file is skipped per-target if already
present (unless --force).

Run:
    modal run modal_scripts/modal_download_medmae.py
    modal run modal_scripts/modal_download_medmae.py --force   # re-download / overwrite
"""

import modal

DEST_SUBDIR = "pretrained"
MIN_OK_BYTES = 50 * 1024 * 1024   # a real ViT-B checkpoint is ~330 MB; anything much
                                  # smaller is a truncated/failed download -> re-fetch

# (google-drive file id, destination filename) — ids from the medical_mae README.
FILES = [
    ("10wqOFCkhyWp6JdSFADrH6Xu9e1am3gXJ", "vit-b_CXR_0.5M_mae.pth"),           # raw MAE
    ("1eZXcoeMJAVjVJUNio2tCyHgiegaa-Vqr", "vit-b_CXR_0.5M_mae_nih14.pth"),     # NIH-14
    ("1yhU-648h5r8wvXGqehZykDoPiXDWHqpj", "vit-b_CXR_0.5M_mae_chexpert.pth"),  # CheXpert
]

# (volume name, mount point). The small volume is always present; the native volume
# is attached only if it already exists (see _existing_targets below).
CANDIDATE_TARGETS = [
    ("chexpert-data",        "/data"),
    ("chexpert-native-data", "/data_native"),
]


def _existing_targets():
    """Keep only the volumes that ALREADY exist (never create the native one just
    for a checkpoint). The small volume is expected; the native one exists once a
    768x640 run has used it."""
    targets = []
    for name, mount in CANDIDATE_TARGETS:
        try:
            vol = modal.Volume.from_name(name, create_if_missing=False)
            targets.append((name, mount, vol))
        except Exception:
            print(f"[skip-volume] {name} does not exist yet — not a download target")
    return targets


app = modal.App("medmae-download")
image = modal.Image.debian_slim(python_version="3.11").pip_install("gdown")

_TARGETS = _existing_targets()
_VOLUMES = {mount: vol for _, mount, vol in _TARGETS}
_MOUNTS = [(name, mount) for name, mount, _ in _TARGETS]


@app.function(image=image, volumes=_VOLUMES, timeout=2 * 3600)
def download(force: bool = False):
    import os
    import gdown

    for name, mount in _MOUNTS:
        dest = f"{mount}/{DEST_SUBDIR}"
        os.makedirs(dest, exist_ok=True)
        print("=" * 70)
        print(f"[volume] {name}  ->  {dest}")
        for file_id, fname in FILES:
            out = f"{dest}/{fname}"
            # already-downloaded detection: present AND a plausible full size (a
            # truncated file re-downloads). --force overrides and re-fetches always.
            if os.path.exists(out) and not force:
                sz = os.path.getsize(out)
                if sz >= MIN_OK_BYTES:
                    print(f"  [skip] {fname} already present ({sz / 1e6:.1f} MB) — --force to redo")
                    continue
                print(f"  [redo] {fname} present but only {sz / 1e6:.1f} MB "
                      f"(< {MIN_OK_BYTES / 1e6:.0f} MB) — re-downloading")
            print(f"  [get ] {fname}  (id={file_id})")
            gdown.download(id=file_id, output=out, quiet=False)
            mb = os.path.getsize(out) / 1e6
            print(f"  [done] {fname}  ->  {out}  ({mb:.1f} MB)")
        _VOLUMES[mount].commit()               # persist this volume's downloads
        print(f"  [{name}] {DEST_SUBDIR}/ now: {sorted(os.listdir(dest))}")


@app.local_entrypoint()
def main(force: bool = False):
    if not _MOUNTS:
        raise SystemExit("no target volumes exist — create chexpert-data first.")
    print(f"[medmae] download targets: {[n for n, _ in _MOUNTS]}")
    download.remote(force)
