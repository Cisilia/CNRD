import argparse
import json
import os
from typing import Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from ced_loader import load_ced_texts_and_labels


def load_vocab_weights(vocab_weights_path: Optional[str]) -> Optional[Dict[str, float]]:
	if not vocab_weights_path:
		return None
	with open(vocab_weights_path, "r", encoding="utf-8") as f:
		data = json.load(f)
		# Expect mapping token -> weight
		return {str(k): float(v) for k, v in data.items()}


def compute_tfidf_features(
	texts,
	max_features: Optional[int],
	ngram_range=(1, 2),
	min_df: int = 2,
	max_df: float = 0.95,
	vocab_weights: Optional[Dict[str, float]] = None,
):
	vectorizer = TfidfVectorizer(
		max_features=max_features,
		ngram_range=ngram_range,
		min_df=min_df,
		max_df=max_df,
		norm="l2",
		use_idf=True,
		smooth_idf=True,
		sublinear_tf=True,
	)
	X = vectorizer.fit_transform(texts)
	feature_names = np.array(vectorizer.get_feature_names_out())

	if vocab_weights:
		weights = np.ones(len(feature_names), dtype=np.float32)
		for i, token in enumerate(feature_names):
			if token in vocab_weights:
				weights[i] = float(vocab_weights[token])
		# Scale columns by weights
		X = X @ np.diag(weights)

	return X, feature_names, vectorizer


def main():
	parser = argparse.ArgumentParser(description="Compute improved TF-IDF features for CED dataset")
	parser.add_argument("--ced_dir", type=str, default="/workspace/CED_Dataset", help="Path to CED_Dataset directory")
	parser.add_argument("--out_dir", type=str, default="/workspace/outputs/tfidf", help="Directory to save features")
	parser.add_argument("--max_features", type=int, default=50000)
	parser.add_argument("--min_df", type=int, default=2)
	parser.add_argument("--max_df", type=float, default=0.95)
	parser.add_argument("--ngram_max", type=int, default=2)
	parser.add_argument("--vocab_weights", type=str, default=None, help="JSON mapping token->weight")
	args = parser.parse_args()

	texts, labels, ids = load_ced_texts_and_labels(args.ced_dir)
	vocab_weights = load_vocab_weights(args.vocab_weights)
	X, feature_names, vectorizer = compute_tfidf_features(
		texts,
		max_features=args.max_features,
		ngram_range=(1, args.ngram_max),
		min_df=args.min_df,
		max_df=args.max_df,
		vocab_weights=vocab_weights,
	)

	os.makedirs(args.out_dir, exist_ok=True)
	# Save sparse matrix in .npz
	npz_path = os.path.join(args.out_dir, "tfidf_features.npz")
	from scipy import sparse
	sparse.save_npz(npz_path, X)

	# Save labels and ids
	with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
		json.dump({"ids": ids, "labels": labels}, f, ensure_ascii=False)

	# Save feature names
	with open(os.path.join(args.out_dir, "feature_names.json"), "w", encoding="utf-8") as f:
		json.dump(feature_names.tolist(), f, ensure_ascii=False)

	# Save vectorizer for reuse
	import joblib
	joblib.dump(vectorizer, os.path.join(args.out_dir, "tfidf_vectorizer.joblib"))

	print(f"Saved TF-IDF features to {npz_path}")


if __name__ == "__main__":
	main()

