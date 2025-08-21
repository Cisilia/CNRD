import json
import os
from typing import List, Tuple


def _list_stem_names(directory_path: str) -> set:
	"""Return a set of file stems (without extension) in a directory."""
	return {
		os.path.splitext(file_name)[0]
		for file_name in os.listdir(directory_path)
		if file_name.endswith(".json")
	}


def load_ced_texts_and_labels(base_dir: str) -> Tuple[List[str], List[int], List[str]]:
	"""
	Load texts and labels from the CED dataset.

	Label mapping:
	- rumor -> 1 (files present in `rumor-repost`)
	- non-rumor -> 0 (files present in `non-rumor-repost`)

	It matches files in `original-microblog` by filename stem against
	`rumor-repost` and `non-rumor-repost`.

	Returns:
	- texts: list of original microblog texts
	- labels: list of integers (1 for rumor, 0 for non-rumor)
	- ids: list of file stems corresponding to each text
	"""
	original_dir = os.path.join(base_dir, "original-microblog")
	rumor_dir = os.path.join(base_dir, "rumor-repost")
	non_rumor_dir = os.path.join(base_dir, "non-rumor-repost")

	if not os.path.isdir(original_dir):
		raise FileNotFoundError(f"Original microblog directory not found: {original_dir}")
	if not os.path.isdir(rumor_dir):
		raise FileNotFoundError(f"Rumor repost directory not found: {rumor_dir}")
	if not os.path.isdir(non_rumor_dir):
		raise FileNotFoundError(f"Non-rumor repost directory not found: {non_rumor_dir}")

	rumor_stems = _list_stem_names(rumor_dir)
	non_rumor_stems = _list_stem_names(non_rumor_dir)

	texts: List[str] = []
	labels: List[int] = []
	ids: List[str] = []

	for file_name in os.listdir(original_dir):
		if not file_name.endswith(".json"):
			continue
		stem = os.path.splitext(file_name)[0]
		label: int
		if stem in rumor_stems:
			label = 1
		elif stem in non_rumor_stems:
			label = 0
		else:
			# Skip originals without a matching label source
			continue

		file_path = os.path.join(original_dir, file_name)
		with open(file_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		text = data.get("text", "")
		if not isinstance(text, str):
			text = str(text)

		texts.append(text)
		labels.append(label)
		ids.append(stem)

	return texts, labels, ids

