import argparse
import os
import re
import shutil

import torch

from model import normalize_drive_id


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def patch_model_file(src_path: str, dst_path: str, drive_id: str) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        text = f.read()

    replacement = f'DEFAULT_GOOGLE_DRIVE_FILE_ID = os.environ.get("DA6401_WEIGHT_FILE_ID", "{drive_id}")'
    text = re.sub(
        r'DEFAULT_GOOGLE_DRIVE_FILE_ID = os\.environ\.get\("DA6401_WEIGHT_FILE_ID",\s*"[^"]*"\)',
        replacement,
        text,
    )

    with open(dst_path, "w", encoding="utf-8") as f:
        f.write(text)


def inspect_checkpoint(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing checkpoint: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must contain model_state_dict.")

    if checkpoint.get("src_vocab") is None or checkpoint.get("tgt_vocab") is None:
        raise ValueError("Checkpoint must contain src_vocab and tgt_vocab. Retrain with the latest code.")

    print("checkpoint_epoch:", checkpoint.get("epoch"))
    print("checkpoint_best_bleu:", checkpoint.get("best_bleu"))
    print("checkpoint_has_vocab:", True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drive-id", required=True, help="Google Drive file id or share URL for transformer_best.pt")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "submission"))
    args = parser.parse_args()

    drive_id = normalize_drive_id(args.drive_id)
    if not drive_id:
        raise ValueError("Drive id is empty.")

    checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "transformer_best.pt")
    inspect_checkpoint(checkpoint_path)

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    patch_model_file(os.path.join(BASE_DIR, "model.py"), os.path.join(args.out, "model.py"), drive_id)

    for name in ["scheduler.py", "lr_scheduler.py"]:
        shutil.copy2(os.path.join(BASE_DIR, name), os.path.join(args.out, name))

    shutil.copytree(os.path.join(BASE_DIR, "vocab"), os.path.join(args.out, "vocab"))

    with open(os.path.join(args.out, "README_SUBMISSION.txt"), "w", encoding="utf-8") as f:
        f.write(
            "Submit the contents of this folder, not the parent folder.\n"
            "Do not include transformer_best.pt directly; model.py downloads it with gdown.\n"
        )

    print("created:", args.out)
    print("drive_file_id:", drive_id)
    print("submit the CONTENTS of this folder")


if __name__ == "__main__":
    main()
