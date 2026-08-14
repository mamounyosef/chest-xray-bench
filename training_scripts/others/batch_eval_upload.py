"""
batch_eval_upload.py  —  put what batch_eval_frontal.py needs onto chexpert-runs.

Per run it uploads:
    results/checkpoints/model.safetensors   the stripped weights from the HF export
    results/thresholds.json                 the frozen per-task thresholds

The stripped weights are used instead of best.pt on purpose: same tensors, about a
third of the bytes (12.4 GB against 39 GB across all runs), and the scorer reads
either. Files already on the volume are skipped unless FORCE is set.

Uploads run in parallel, since each one is a single sequential transfer and the
link is the bottleneck.

Run:  python training_scripts/others/batch_eval_upload.py
"""

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ============================ CONFIG (edit here) ============================
EXPORT = Path(r"D:\chexpert-bench-hf")   # where export_hf.py wrote the weights
WORKERS = 6                              # parallel transfers
FORCE = False                            # re-upload files already on the volume
VOLUME = "chexpert-runs"
# ===========================================================================

PKG_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(PKG_ROOT / "others"))
import batch_eval_frontal as bf          # noqa: E402


def plan_uploads(vol) -> list:
    """[(local path, remote path)] for every file missing on the volume."""
    have = set()
    if not FORCE:
        try:
            have = {e.path.lstrip("/") for e in vol.listdir("/", recursive=True)}
        except Exception as exc:
            print(f"could not list the volume ({exc}); uploading everything")

    import shared_code as sc

    jobs = []
    for name, _mp in bf.discover_runs():
        # two-stage runs keep the CheXpert model in a subfolder, and that is where
        # the scorer looks, so the weights have to land there and not flat
        sub = sc.finetune_ckpt_subdir(sc.load_config(PKG_ROOT / name, verbose=False))
        ckpt_dir = f"{name}/results/checkpoints" + (f"/{sub}" if sub else "")
        pairs = [
            (EXPORT / name / "model.safetensors", f"{ckpt_dir}/model.safetensors"),
            (PKG_ROOT / name / "results" / "thresholds.json",
             f"{name}/results/thresholds.json"),
        ]
        for local, remote in pairs:
            if not local.exists():
                continue
            if remote in have:
                continue
            jobs.append((local, remote))
    return jobs


def main():
    import modal

    vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
    jobs = plan_uploads(vol)
    total_gb = sum(p.stat().st_size for p, _ in jobs) / 2**30
    print(f"{len(jobs)} files to upload, {total_gb:.1f} GB, {WORKERS} at a time\n")
    if not jobs:
        print("nothing to do, the volume already has everything")
        return

    def put(job):
        local, remote = job
        with vol.batch_upload(force=True) as batch:
            batch.put_file(str(local), remote)
        return remote, local.stat().st_size / 2**20

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(put, j): j for j in jobs}
        for fut in as_completed(futures):
            local, remote = futures[fut]
            done += 1
            try:
                _, mb = fut.result()
                print(f"[{done}/{len(jobs)}] {remote}  ({mb:.0f} MB)")
            except Exception as exc:
                print(f"[{done}/{len(jobs)}] FAILED {remote}: {exc!r}")

    print("\nupload finished")


if __name__ == "__main__":
    main()
