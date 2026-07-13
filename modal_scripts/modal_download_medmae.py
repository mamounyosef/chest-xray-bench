r"""
modal_download_medmae.py  —  download the Medical-MAE ViT-B/16 checkpoints
(lambert-x/medical_mae) straight onto the chexpert-data Modal volume, so training
reads them locally on Modal and they never touch your PC.

Downloads 3 files by Google-Drive file id into  /data/pretrained/ :
    vit-b_CXR_0.5M_mae.pth            <- Run 1  (raw MAE self-supervised backbone)
    vit-b_CXR_0.5M_mae_nih14.pth      <- Run 2  (NIH ChestX-ray14 fine-tuned, mAUC 83.0)
    vit-b_CXR_0.5M_mae_chexpert.pth   <- Run 3  (CheXpert fine-tuned, mAUC 89.3)

The 3 run configs point model.checkpoint_path at these exact paths.

Run:
    modal run modal_scripts/modal_download_medmae.py
    modal run modal_scripts/modal_download_medmae.py --force   # re-download / overwrite
"""

import modal

VOLUME = "chexpert-data"
MOUNT = "/data"
DEST_SUBDIR = "pretrained"

# (google-drive file id, destination filename) — ids from the medical_mae README.
FILES = [
    ("10wqOFCkhyWp6JdSFADrH6Xu9e1am3gXJ", "vit-b_CXR_0.5M_mae.pth"),           # raw MAE
    ("1eZXcoeMJAVjVJUNio2tCyHgiegaa-Vqr", "vit-b_CXR_0.5M_mae_nih14.pth"),     # NIH-14
    ("1yhU-648h5r8wvXGqehZykDoPiXDWHqpj", "vit-b_CXR_0.5M_mae_chexpert.pth"),  # CheXpert
]

app = modal.App("medmae-download")
image = modal.Image.debian_slim(python_version="3.11").pip_install("gdown")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)


@app.function(image=image, volumes={MOUNT: vol}, timeout=2 * 3600)
def download(force: bool = False):
    import os
    import gdown

    dest = f"{MOUNT}/{DEST_SUBDIR}"
    os.makedirs(dest, exist_ok=True)
    vol.reload()

    for file_id, name in FILES:
        out = f"{dest}/{name}"
        if os.path.exists(out) and not force:
            mb = os.path.getsize(out) / 1e6
            print(f"[skip] {name} already present ({mb:.1f} MB) — use --force to redo")
            continue
        print(f"[get ] {name}  (id={file_id})")
        gdown.download(id=file_id, output=out, quiet=False)
        mb = os.path.getsize(out) / 1e6
        print(f"[done] {name}  ->  {out}  ({mb:.1f} MB)")

    vol.commit()                          # persist the downloads on the volume
    print(f"\n[medmae] /data/{DEST_SUBDIR} now: {sorted(os.listdir(dest))}")


@app.local_entrypoint()
def main(force: bool = False):
    download.remote(force)
