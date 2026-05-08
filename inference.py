"""
inference.py - Standalone inference script
DA6401 Assignment 3

Usage:
    python inference.py --sentence "Ein Hund spielt im Garten."
    python inference.py --interactive
"""

import argparse
import torch
from model import Transformer


def translate(german_sentence: str, device: str = "cpu") -> str:
    model = Transformer().to(device)
    model.eval()
    return model.infer(german_sentence)


def main():
    parser = argparse.ArgumentParser(description="German → English Translator")
    parser.add_argument("--sentence",    type=str, default=None,
                        help="German sentence to translate")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive mode: translate sentences from stdin")
    parser.add_argument("--device",      type=str, default="cpu")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print("[Inference] Loading model …")
    model = Transformer().to(device)
    model.eval()
    print("[Inference] Model ready.")

    if args.interactive:
        print("Enter German sentences (Ctrl+C to quit):")
        try:
            while True:
                sent = input("DE > ").strip()
                if sent:
                    translation = model.infer(sent)
                    print(f"EN > {translation}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
    elif args.sentence:
        translation = model.infer(args.sentence)
        print(f"Input:  {args.sentence}")
        print(f"Output: {translation}")
    else:
        # Demo sentences
        demo = [
            "Ein Hund spielt im Garten.",
            "Zwei Männer stehen vor einem Gebäude.",
            "Ein Kind läuft auf der Straße.",
        ]
        for sent in demo:
            out = model.infer(sent)
            print(f"DE: {sent}")
            print(f"EN: {out}\n")


if __name__ == "__main__":
    main()
