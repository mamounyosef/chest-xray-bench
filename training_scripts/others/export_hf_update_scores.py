"""
export_hf_update_scores.py  —  refresh the published config.json scores.

Each model's config.json on the Hub carries a "results" block. This rewrites that
block from the frontal-only evaluations (202 valid200 images, 518 test500) and
uploads just the config files, leaving the weights untouched.

Run:  python training_scripts/others/export_hf_update_scores.py
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ============================ CONFIG (edit here) ============================
EXPORT = Path(r"D:\chexpert-bench-hf")
REPO = "mamounyosef/chest-xray-bench"
WORKERS = 4
# ===========================================================================

RUNS = Path(__file__).resolve().parent.parent


def frontal(run: str, split: str):
    p = RUNS / run / "results" / f"{split}_frontal_results.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return {
        "n_images": d["n_images"],
        "mean_auroc": round(d["macro"]["mean_auroc"], 4),
        "per_task_auroc": {k: round(v["auroc"], 4)
                           for k, v in d["per_task"].items()},
    }


def refresh(folder: Path) -> bool:
    cfg_path = folder / "config.json"
    if not cfg_path.exists():
        return False
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    run = cfg["run"]
    res = {}
    for split in ("valid200", "test500"):
        v = frontal(run, split)
        if v:
            res[split] = v
    if not res:
        print(f"  no frontal results for {run}")
        return False
    cfg["results"] = res
    cfg["evaluation"] = ("frontal views only: 202 of the 234 valid200 images and "
                         "518 of the 668 test500 images, matching the training data")
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return True


def main():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN first")
    api = HfApi(token=token)

    folders = sorted(p for p in EXPORT.iterdir()
                     if p.is_dir() and (p / "config.json").exists())
    updated = [p for p in folders if refresh(p)]
    print(f"{len(updated)} of {len(folders)} config files refreshed, uploading\n")

    def put(folder: Path):
        api.upload_file(path_or_fileobj=str(folder / "config.json"),
                        path_in_repo=f"{folder.name}/config.json",
                        repo_id=REPO, repo_type="model",
                        commit_message=f"Frontal-only scores for {folder.name}")
        return folder.name

    done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = {pool.submit(put, f): f for f in updated}
        for fut in as_completed(futs):
            done += 1
            try:
                print(f"[{done}/{len(updated)}] {fut.result()}")
            except Exception as exc:
                print(f"[{done}/{len(updated)}] FAILED {futs[fut].name}: {exc!r}")


if __name__ == "__main__":
    main()
