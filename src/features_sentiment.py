import argparse
import json
import os
from typing import List, Tuple

import numpy as np
from ced_loader import load_ced_texts_and_labels


def compute_sentiment_scores(texts: List[str]) -> np.ndarray:
	"""
	Compute sentiment scores for Chinese texts.

	We use HuggingFace transformers with a Chinese sentiment model.
	Outputs a 2D array of shape [N, 2] with [neg_prob, pos_prob].
	"""
	from transformers import AutoTokenizer, AutoModelForSequenceClassification
	import torch

	model_id = "uer/roberta-base-finetuned-jd-binary-chinese"
	tokenizer = AutoTokenizer.from_pretrained(model_id)
	model = AutoModelForSequenceClassification.from_pretrained(model_id)
	model.eval()

	all_scores: List[List[float]] = []
	batch_size = 32
	for start in range(0, len(texts), batch_size):
		batch = texts[start:start + batch_size]
		inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt")
		with torch.no_grad():
			outputs = model(**inputs)
			logits = outputs.logits
			probs = torch.softmax(logits, dim=-1)
		all_scores.extend(probs.cpu().numpy().tolist())

	return np.array(all_scores, dtype=np.float32)


def main():
	parser = argparse.ArgumentParser(description="Compute sentiment features for CED dataset")
	parser.add_argument("--ced_dir", type=str, default="/workspace/CED_Dataset")
	parser.add_argument("--out_dir", type=str, default="/workspace/outputs/sentiment")
	args = parser.parse_args()

	texts, labels, ids = load_ced_texts_and_labels(args.ced_dir)
	scores = compute_sentiment_scores(texts)

	os.makedirs(args.out_dir, exist_ok=True)
	# Save as npy
	import numpy as np
	np.save(os.path.join(args.out_dir, "sentiment.npy"), scores)
	with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
		json.dump({"ids": ids, "labels": labels}, f, ensure_ascii=False)
	print(f"Saved sentiment scores to {os.path.join(args.out_dir, 'sentiment.npy')}")


if __name__ == "__main__":
	main()

