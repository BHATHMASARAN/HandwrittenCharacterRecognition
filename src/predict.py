import argparse
import json

from src.inference.predictor import CharacterPredictor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a handwritten character image")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--word", action="store_true", help="Segment and predict a full word/sentence")
    parser.add_argument("--model", type=str, default=None, help="Override checkpoint path")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.image_path, "rb") as f:
        image_bytes = f.read()

    predictor = CharacterPredictor(model_path=args.model)

    if args.word:
        result = predictor.predict_word(image_bytes)
    else:
        result = predictor.predict(image_bytes)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
