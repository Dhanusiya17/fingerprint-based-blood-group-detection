#!/usr/bin/env python3
from pathlib import Path
import argparse

from app.ml.train import train_and_save


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", default="resnet50", choices=["resnet50", "vgg16", "mobilenet_v2", "lenet"]) 
	parser.add_argument("--out", default=None)
	args = parser.parse_args()
	out_dir = Path(args.out) if args.out else None
	result = train_and_save(model_name=args.model, output_dir=out_dir)
	print(result)


if __name__ == "__main__":
	main()