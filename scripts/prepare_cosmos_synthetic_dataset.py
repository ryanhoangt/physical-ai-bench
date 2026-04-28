#!/usr/bin/env python3
"""
Prepare dataset files for physical-ai-bench evaluation from cosmos synthetic data.

For each cosmos subfolder this script:
  1. Finds valid .mp4 files (those with a matching frame folder, filtering out extras)
  2. Copies frame_000000.jpg alongside the video as a condition image {task_name}.jpg
  3. Writes datasets/cosmos_synthetic_data/{subset}/full_info.json

Usage:
  # Process all 6 subsets
  python scripts/prepare_cosmos_synthetic_dataset.py

  # Process a single subset
  python scripts/prepare_cosmos_synthetic_dataset.py --subset cosmos_predict2_14b_gr1_behavior

Eval command after running this script (example for one subset):
  cd generation
  uv run python -m torch.distributed.run --standalone --nproc_per_node 1 evaluate.py \\
    --mode custom_input \\
    --prompt_file ../datasets/cosmos_synthetic_data/cosmos_predict2_14b_gr1_behavior/full_info.json \\
    --custom_image_folder ../datasets/cosmos_synthetic_data/cosmos_predict2_14b_gr1_behavior/condition_image \\
    --videos_path ../cosmos_synthetic_data/cosmos_predict2_14b_gr1_behavior \\
    --dimension aesthetic_quality background_consistency imaging_quality motion_smoothness \\
               overall_consistency subject_consistency i2v_background i2v_subject \\
    --output_path ../evaluation_results/cosmos_predict2_14b_gr1_behavior
"""

import argparse
import json
import shutil
from pathlib import Path

DIMENSIONS = [
    "aesthetic_quality",
    "background_consistency",
    "imaging_quality",
    "motion_smoothness",
    "overall_consistency",
    "subject_consistency",
    "i2v_background",
    "i2v_subject",
]

VALID_SUBSETS = [
    "cosmos_predict2_14b_gr1_behavior",
    "cosmos_predict2_14b_gr1_env",
    "cosmos_predict2_14b_gr1_object",
    "cosmos_predict2_2b_gr1_behavior",
    "cosmos_predict2_2b_gr1_env",
    "cosmos_predict2_2b_gr1_object",
    "low",
    "high",
]


def parse_section_from_txt(text: str, section: str) -> str:
    """Extract a named [Section] from .txt metadata content."""
    lines = text.splitlines()
    result_lines = []
    in_section = False
    for line in lines:
        if line.strip() == f"[{section}]":
            in_section = True
            continue
        if in_section:
            if line.strip().startswith("[") and line.strip().endswith("]"):
                break
            result_lines.append(line)
    return "\n".join(result_lines).strip()


def parse_prompt_from_txt(txt_path: Path) -> str:
    """Extract prompt from .txt: prefer [Refined Prompt], fall back to [Prompt]."""
    text = txt_path.read_text(encoding="utf-8")
    refined = parse_section_from_txt(text, "Refined Prompt")
    if refined:
        return refined
    return parse_section_from_txt(text, "Prompt")


def process_subset(subset_name: str, cosmos_data_dir: Path, output_dir: Path) -> int:
    subset_dir = cosmos_data_dir / subset_name
    if not subset_dir.exists():
        print(f"  [{subset_name}] directory not found, skipping")
        return 0

    out_dir = output_dir / subset_name
    condition_image_dir = out_dir / "condition_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    condition_image_dir.mkdir(parents=True, exist_ok=True)

    full_info = []
    skipped = 0

    # Sort by numeric index prefix so full_info.json is in a natural order
    mp4_files = sorted(
        subset_dir.glob("*.mp4"),
        key=lambda p: int(p.stem.split("_")[0]) if p.stem.split("_")[0].isdigit() else float("inf"),
    )

    for mp4_path in mp4_files:
        task_name = mp4_path.stem  # e.g. "0_Open the box"
        frame_dir = subset_dir / task_name

        # Only valid videos have a matching frame folder; extras don't
        if not frame_dir.is_dir():
            skipped += 1
            continue

        first_frame = frame_dir / "frame_000000.jpg"
        if not first_frame.exists():
            print(f"  WARNING: frame_000000.jpg missing in {frame_dir.name}, skipping")
            skipped += 1
            continue

        # video_id uses robot_ prefix so evaluate_vqa.py maps it to the 'robot' category
        video_id = f"robot_{task_name}"

        # Symlink robot_{task_name}.mp4 -> {task_name}.mp4 so eval scripts can find it by video_id
        symlink = subset_dir / f"{video_id}.mp4"
        if not symlink.exists() and not symlink.is_symlink():
            symlink.symlink_to(mp4_path.name)

        # Copy first frame into datasets/{subset}/condition_image/ using the video_id name
        condition_image = condition_image_dir / f"{video_id}.jpg"
        if not condition_image.exists():
            shutil.copy2(first_frame, condition_image)

        # Parse prompt from .txt; fall back to task name if .txt is missing
        txt_path = subset_dir / f"{task_name}.txt"
        if txt_path.exists():
            prompt = parse_prompt_from_txt(txt_path)
            if not prompt:
                prompt = task_name.split("_", 1)[-1]
        else:
            prompt = task_name.split("_", 1)[-1]

        full_info.append({
            "video_id": video_id,
            "task_name": task_name,   # original name used to locate frame dirs and .txt files
            "prompt_en": prompt,
            "dimension": DIMENSIONS,
            "image_name": f"{video_id}.jpg",
            "reference_video": f"{video_id}.mp4",
        })

    out_json = out_dir / "full_info.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_info, f, indent=2, ensure_ascii=False)

    print(f"  [{subset_name}] {len(full_info)} entries written, {skipped} skipped -> {out_json}")
    return len(full_info)


def main():
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Prepare cosmos synthetic dataset files for physical-ai-bench eval"
    )
    parser.add_argument(
        "--cosmos_data_dir",
        type=Path,
        default=repo_root / "cosmos_synthetic_data",
        help="Root directory containing cosmos synthetic subfolders (default: repo/cosmos_synthetic_data)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=repo_root / "datasets" / "cosmos_synthetic_data",
        help="Output directory for full_info.json files (default: repo/datasets/cosmos_synthetic_data)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        choices=VALID_SUBSETS,
        metavar="SUBSET",
        help=(
            "Process a single subset. One of: "
            + ", ".join(VALID_SUBSETS)
            + ". If omitted, all 6 subsets are processed."
        ),
    )
    args = parser.parse_args()

    subsets = [args.subset] if args.subset else VALID_SUBSETS

    print(f"cosmos_data_dir : {args.cosmos_data_dir}")
    print(f"output_dir      : {args.output_dir}")
    print()

    total = 0
    for subset in subsets:
        total += process_subset(subset, args.cosmos_data_dir, args.output_dir)

    print(f"\nTotal entries written: {total}")


if __name__ == "__main__":
    main()
