import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import DEFAULT_GOOGLE_DRIVE_FILE_ID, Transformer


def main():
    model = Transformer()
    print("weights_loaded:", model.weights_loaded)
    print("drive_file_id_set:", bool(DEFAULT_GOOGLE_DRIVE_FILE_ID))
    print("src_vocab_size:", len(model.src_vocab) if model.src_vocab is not None else None)
    print("tgt_vocab_size:", len(model.tgt_vocab) if model.tgt_vocab is not None else None)

    sentence = "ein mann in einem roten hemd spielt gitarre ."
    print("input:", sentence)
    print("output:", model.infer(sentence, max_len=60))

    if not model.weights_loaded:
        raise SystemExit(
            "ERROR: trained weights were not loaded. "
            "Upload transformer_best.pt to Google Drive and set DEFAULT_GOOGLE_DRIVE_FILE_ID."
        )


if __name__ == "__main__":
    main()
