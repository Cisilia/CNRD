import argparse
import os
import json
from typing import List

import numpy as np
from ced_loader import load_ced_texts_and_labels


def get_text_embedding_batch(texts: List[str], model_name: str) -> np.ndarray:
	"""
	Compute sentence embeddings for a list of texts using sentence-transformers.
	"""
	from sentence_transformers import SentenceTransformer
	model = SentenceTransformer(model_name)
	embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
	return embeddings


def main():
	parser = argparse.ArgumentParser(description="Compute sentence embeddings for CED dataset")
	parser.add_argument("--ced_dir", type=str, default="/workspace/CED_Dataset")
	parser.add_argument("--out_dir", type=str, default="/workspace/outputs/embeddings")
	parser.add_argument("--model", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", help="HuggingFace model id for sentence-transformers")
	args = parser.parse_args()

	texts, labels, ids = load_ced_texts_and_labels(args.ced_dir)
	embeddings = get_text_embedding_batch(texts, args.model)

	os.makedirs(args.out_dir, exist_ok=True)
	np.save(os.path.join(args.out_dir, "embeddings.npy"), embeddings)
	with open(os.path.join(args.out_dir, "labels.json"), "w", encoding="utf-8") as f:
		json.dump({"ids": ids, "labels": labels}, f, ensure_ascii=False)
	print(f"Saved embeddings to {os.path.join(args.out_dir, 'embeddings.npy')}")


if __name__ == "__main__":
	main()

