#!/usr/bin/env python3
import os
import sys
import json
import csv
import math
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Any

# This script extracts propagation features for each cascade in CED_Dataset.
# It produces one CSV row per original microblog (rumor and non-rumor), with features:
# - id, token, user_id_from_filename, label (rumor/non_rumor)
# - root_time_unix
# - root_likes, root_comments, root_reposts
# - cascade_size
# - forward_count_estimated (nodes with empty parent OR parent not found)
# - propagation_depth
# - depth_mean, depth_median, max_breadth
# - time_to_root_min_s, time_to_root_median_s, time_to_root_mean_s, time_to_root_max_s
# - virality_score_mean_depth (proxy structural virality)
# - repost_file_found (bool)
#
# Note on dataset:
# - CED repost files do not distinguish repost vs comment and may have incomplete parent links.
# - We define root depth = 0.
# - Nodes with parent == "" or parent not present in this file are treated as direct children (depth = 1).
# - Virality score is approximated as mean depth across nodes.

DATASET_DIR = os.path.join(os.path.dirname(__file__), "CED_Dataset")
ORIG_DIR = os.path.join(DATASET_DIR, "original-microblog")
RUMOR_REPOST_DIR = os.path.join(DATASET_DIR, "rumor-repost")
NONRUMOR_REPOST_DIR = os.path.join(DATASET_DIR, "non-rumor-repost")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "propagation_features.csv")


def parse_filename_components(filename: str) -> Tuple[str, str, str]:
	# Filename format: "<id>_<token>_<userId>.json"
	stem = filename
	if stem.endswith(".json"):
		stem = stem[:-5]
	parts = stem.split("_")
	if len(parts) < 3:
		return parts[0], parts[1] if len(parts) > 1 else "", ""
	return parts[0], parts[1], parts[2]


def safe_int(value: Any, default: int = 0) -> int:
	try:
		return int(value)
	except Exception:
		return default


def safe_float(value: Any, default: float = 0.0) -> float:
	try:
		return float(value)
	except Exception:
		return default


def parse_repost_file(path: str) -> List[Dict[str, Any]]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
			if isinstance(data, list):
				return data
			return []
	except Exception:
		return []


def parse_original_file(path: str) -> Dict[str, Any]:
	try:
		with open(path, "r", encoding="utf-8") as f:
			data = json.load(f)
			if isinstance(data, dict):
				return data
			return {}
	except Exception:
		return {}


def try_parse_date(date_str: str) -> datetime:
	# CED date format example: "2013-03-27 07:38:09"
	try:
		dt_naive = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
		# Treat as UTC to avoid local timezone dependence; only diffs matter
		return dt_naive.replace(tzinfo=timezone.utc)
	except Exception:
		return None


def compute_depths(nodes: List[Dict[str, Any]]) -> Tuple[Dict[str, int], int, float, float, int, int]:
	# Returns (mid_to_depth, max_depth, mean_depth, median_depth, max_breadth, forward_count_estimated)
	mid_to_node: Dict[str, Dict[str, Any]] = {}
	children_of: Dict[str, List[str]] = {}
	parent_of: Dict[str, str] = {}
	candidate_direct_children: int = 0

	for node in nodes:
		mid = node.get("mid")
		if not mid:
			continue
		mid_to_node[mid] = node
		kids = node.get("kids") or []
		if isinstance(kids, list):
			for child_mid in kids:
				children_of.setdefault(mid, []).append(child_mid)
				parent_of[child_mid] = mid
		parent = node.get("parent", "")
		if parent:
			# If parent not in file, we cannot set children list here, but we record parent link
			parent_of[mid] = parent
		else:
			candidate_direct_children += 1

	# Assign depths using iterative relaxation since some parents may be missing; 
	# Nodes whose parent is not in file or parent == "" get depth=1.
	mid_to_depth: Dict[str, int] = {}
	queue: List[str] = []
	for mid, node in mid_to_node.items():
		p = parent_of.get(mid, "")
		if (not p) or (p not in mid_to_node):
			mid_to_depth[mid] = 1
			queue.append(mid)

	# BFS/DP to fill deeper nodes
	# We will iterate over edges derived from explicit parent links and from kids
	changed = True
	while changed:
		changed = False
		for mid, node in mid_to_node.items():
			p = parent_of.get(mid)
			if p in mid_to_depth and mid_to_depth.get(mid) is None:
				mid_to_depth[mid] = mid_to_depth[p] + 1
				changed = True
			# Also check kids-defined edges
			for child in children_of.get(mid, []):
				if mid in mid_to_depth and child in mid_to_node and child not in mid_to_depth:
					mid_to_depth[child] = mid_to_depth[mid] + 1
					changed = True

	# For any remaining without depth (cycles or unresolved), set depth=1 conservatively
	for mid in mid_to_node.keys():
		if mid not in mid_to_depth:
			mid_to_depth[mid] = 1

	depth_values = list(mid_to_depth.values())
	max_depth = max(depth_values) if depth_values else 0
	mean_depth = sum(depth_values) / len(depth_values) if depth_values else 0.0
	median_depth = 0.0
	if depth_values:
		s = sorted(depth_values)
		m = len(s)
		median_depth = s[m // 2] if m % 2 == 1 else (s[m // 2 - 1] + s[m // 2]) / 2.0

	# breadth by depth
	breadth: Dict[int, int] = {}
	for d in depth_values:
		breadth[d] = breadth.get(d, 0) + 1
	max_breadth = max(breadth.values()) if breadth else 0

	# forward_count_estimated: nodes with no parent or parent missing in file
	forward_count_estimated = 0
	for mid, node in mid_to_node.items():
		p = parent_of.get(mid, "")
		if (not p) or (p not in mid_to_node):
			forward_count_estimated += 1

	return mid_to_depth, max_depth, mean_depth, median_depth, max_breadth, forward_count_estimated


def compute_time_deltas_seconds(nodes: List[Dict[str, Any]], root_time_unix: int) -> Tuple[float, float, float, float]:
	if not nodes:
		return 0.0, 0.0, 0.0, 0.0
	root_dt = datetime.fromtimestamp(root_time_unix, tz=timezone.utc)
	deltas: List[float] = []
	for node in nodes:
		date_str = node.get("date")
		if not date_str:
			continue
		node_dt = try_parse_date(date_str)
		if not node_dt:
			continue
		delta_sec = (node_dt - root_dt).total_seconds()
		deltas.append(delta_sec)
	if not deltas:
		return 0.0, 0.0, 0.0, 0.0
	min_s = float(min(deltas))
	max_s = float(max(deltas))
	mean_s = float(sum(deltas) / len(deltas))
	med_s = 0.0
	if deltas:
		s = sorted(deltas)
		m = len(s)
		med_s = float(s[m // 2] if m % 2 == 1 else (s[m // 2 - 1] + s[m // 2]) / 2.0)
	return min_s, med_s, mean_s, max_s


def write_csv_header(writer: csv.writer) -> None:
	writer.writerow([
		"id",
		"token",
		"user_id_from_filename",
		"label",
		"root_time_unix",
		"root_likes",
		"root_comments",
		"root_reposts",
		"cascade_size",
		"forward_count_estimated",
		"propagation_depth",
		"depth_mean",
		"depth_median",
		"max_breadth",
		"time_to_root_min_s",
		"time_to_root_median_s",
		"time_to_root_mean_s",
		"time_to_root_max_s",
		"virality_score_mean_depth",
		"repost_file_found"
	])


def extract_features_for_file(filename: str, label: str, writer: csv.writer) -> None:
	orig_path = os.path.join(ORIG_DIR, filename)
	orig = parse_original_file(orig_path)
	id_part, token, user_id_from_filename = parse_filename_components(filename)
	root_time_unix = safe_int(orig.get("time", 0))
	root_likes = safe_int(orig.get("likes", 0))
	root_comments = safe_int(orig.get("comments", 0))
	root_reposts = safe_int(orig.get("reposts", 0))

	# Find matching repost file in rumor or non-rumor dir
	repost_path = None
	for d in (RUMOR_REPOST_DIR, NONRUMOR_REPOST_DIR):
		candidate = os.path.join(d, filename)
		if os.path.exists(candidate):
			repost_path = candidate
			break

	nodes: List[Dict[str, Any]] = []
	repost_found = False
	if repost_path and os.path.exists(repost_path):
		nodes = parse_repost_file(repost_path)
		repost_found = True

	mid_to_depth, max_depth, mean_depth, median_depth, max_breadth, forward_count_estimated = compute_depths(nodes)
	min_s, med_s, mean_s, max_s = compute_time_deltas_seconds(nodes, root_time_unix)
	cascade_size = len(nodes)
	virality_score = mean_depth  # proxy

	writer.writerow([
		id_part,
		token,
		user_id_from_filename,
		label,
		root_time_unix,
		root_likes,
		root_comments,
		root_reposts,
		cascade_size,
		forward_count_estimated,
		max_depth,
		f"{mean_depth:.6f}",
		f"{median_depth:.6f}",
		max_breadth,
		f"{min_s:.6f}",
		f"{med_s:.6f}",
		f"{mean_s:.6f}",
		f"{max_s:.6f}",
		f"{virality_score:.6f}",
		str(repost_found)
	])


def main() -> int:
	if not os.path.isdir(ORIG_DIR):
		print(f"Error: original-microblog directory not found at {ORIG_DIR}", file=sys.stderr)
		return 1

	files = sorted([f for f in os.listdir(ORIG_DIR) if f.endswith('.json')])
	if not files:
		print("No original-microblog JSON files found.", file=sys.stderr)
		return 1

	# Optionally, allow limiting via environment variables
	limit = None
	if "LIMIT" in os.environ:
		try:
			limit = int(os.environ["LIMIT"])
		except Exception:
			limit = None

	with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f_out:
		writer = csv.writer(f_out)
		write_csv_header(writer)
		count = 0
		for filename in files:
			label = "rumor" if os.path.exists(os.path.join(RUMOR_REPOST_DIR, filename)) else (
				"non_rumor" if os.path.exists(os.path.join(NONRUMOR_REPOST_DIR, filename)) else "unknown"
			)
			extract_features_for_file(filename, label, writer)
			count += 1
			if count % 200 == 0:
				print(f"Processed {count} cascades...")
			if limit is not None and count >= limit:
				break

	print(f"Done. Wrote features to {OUTPUT_CSV}")
	return 0


if __name__ == "__main__":
	sys.exit(main())