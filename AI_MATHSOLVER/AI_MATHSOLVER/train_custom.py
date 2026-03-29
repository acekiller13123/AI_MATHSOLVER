"""
Optional layout for custom EasyOCR / recognition training data.

Create pairs:
  training_data/images/<id>.png
  training_data/labels/<id>.txt   (one line = ground-truth text)

Fine-tuning EasyOCR itself is done with their official trainer (see EasyOCR docs).
This app uses pretrained EasyOCR at inference time; custom data prepares you for that pipeline.
"""

from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent / "training_data"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    print(f"Created:\n  {root / 'images'}\n  {root / 'labels'}")
    print("Add matching basenames, e.g. prob_01.png and prob_01.txt")


if __name__ == "__main__":
    main()
