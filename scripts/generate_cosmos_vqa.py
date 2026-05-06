#!/usr/bin/env python3
"""
Generate VQA questions for cosmos synthetic videos using an MLLM via LiteLLM.

For each video in a cosmos synthetic subset, uses frame_000000.jpg + task prompt
to generate yes/no questions across the physical common sense ontology.

Output: datasets/cosmos_synthetic_data/{subset}/vqa/{video_id}.json

Prerequisites:
  - Run prepare_cosmos_synthetic_dataset.py first (needs full_info.json)
  - Set OPENAI_API_KEY (or relevant key) in environment

Usage:
  # Test on 3 videos
  python scripts/generate_cosmos_vqa.py --subset cosmos_predict2_2b_gr1_behavior --n_videos 3

  # Full run
  python scripts/generate_cosmos_vqa.py --subset cosmos_predict2_2b_gr1_behavior

  # Different model
  python scripts/generate_cosmos_vqa.py --subset cosmos_predict2_2b_gr1_behavior --model anthropic/claude-opus-4-7
"""

import argparse
import base64
import json
from pathlib import Path
import os

from litellm import completion, completion_cost

# Full 16-subcategory ontology grouped by main category.
# Note: Space:_Interaction and Space:_Geometry are the robot-benchmark-specific names
# used in existing eval files (cf. ontology's Affordance / Plausibility+Environment).
CATEGORIES = {
    "Space": [
        "Space:_Relationship",
        "Space:_Plausibility",
        "Space:_Interaction",
        "Space:_Environment",
        "Space:_Geometry",
    ],
    "Time": [
        "Time:_Actions",
        "Time:_Order",
        "Time:_Causality",
        "Time:_Camera",
        "Time:_Planning",
    ],
    "Physics": [
        "Physics:_Attributes",
        "Physics:_States",
        "Physics:_Object_Permanence",
        "Physics:_Mechanics",
        "Physics:_Electromagnetism",
        "Physics:_Thermodynamics",
        "Physics:_Anti-Physics",
    ],
}
VALID_CATEGORIES = {cat for cats in CATEGORIES.values() for cat in cats}

SYSTEM_PROMPT = """\
You generate VQA evaluation data for robot manipulation videos.

You will receive the FIRST FRAME of a robot video and the TASK DESCRIPTION (what the robot \
should do). Generate yes/no questions that assess physical common sense compatibility of the \
video, the scene and the manipulation task.

ONTOLOGY — 3 main categories, 16 subcategories total:

Space:
  Space:_Relationship   - Spatial relationships between objects (left/right/on/under/near). \
Perspective matters (left of person vs left of camera).
  Space:_Plausibility   - Whether a spatial relationship or configuration is physically feasible.
  Space:_Interaction    - Robot or subject interaction with objects (grasping, placing, pushing, etc.).
  Space:_Environment    - Scene or surrounding environment (background, surface, room context).
  Space:_Geometry       - Shape, size, spatial geometry of objects in the scene.

Time:
  Time:_Actions         - Robot actions: movement, direction, intensity, task execution, \
success/failure, goal completion.
  Time:_Order           - Sequential order of events (what happens before/after what).
  Time:_Causality       - Whether event A causes event B.
  Time:_Camera          - Camera position, angle, movement, and scene transitions.
  Time:_Planning        - Future plan or next steps inferred from past observations.

Physics:
  Physics:_Attributes        - Physical properties: color, material, size, texture, mass, \
temperature, solidity.
  Physics:_States            - Object state changes (open→closed, raw→cooked, moved, filled→empty).
  Physics:_Object_Permanence - Objects staying visible/present as physically expected; which \
properties can/cannot change.
  Physics:_Mechanics         - Laws of mechanics: statics (balance, stability), kinematics \
(velocity, motion), dynamics (gravity, friction, collision).
  Physics:_Electromagnetism  - Optics (lighting, shadow, reflection, occlusion), electricity, \
magnetism.
  Physics:_Thermodynamics    - Heat, temperature change, evaporation, thermal expansion.
  Physics:_Anti-Physics      - Situations defying physics: anti-gravity, objects disappearing, \
reverse causality.

RULES:
1. Create questions that require both the information in the caption and the robot manipulation \
task with your creativity. Do NOT create any questions with answers that are directly given in the caption.
2. Select at lease 1 subcategory from each main category (Space, Time, Physics). Each subcategory can have \
more than 1 question, especially those in Time and Space category, totaling from 6-10 questions. \
Choose the subcategories most naturally applicable to this specific scene and task.
3. Target ~70% answer "A" (yes) and ~30% answer "B" (no) across all questions.
4. For "B" answers: write plausible but INCORRECT statements as good distractors.
5. Scene questions (Relationship, Geometry, Attributes, Environment): ground the answer in \
what you observe in the image.
6. Task questions (Interaction, Actions, Order, Causality): assume the task completes \
successfully when answering "A".
7. Camera/Planning questions: describe the visible camera setup or infer next steps from the task.
8. Physics questions: focus on observable or task-implied physical phenomena.
9. Questions must be specific, concrete, and unambiguous.

Return ONLY a JSON array — no markdown fences, no explanation:
[
  {{"category": "Space:_Relationship", "question": "...", "answer": "A"}},
  {{"category": "Space:_Interaction",  "question": "...", "answer": "A"}},
  ...
]
"""


# Video IDs used as few-shot ICL examples (from the reference benchmark dataset)
ICL_EXAMPLE_IDS = ["robot_000", "robot_014"]


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_user_content(image_path: Path, task_prompt: str) -> list:
    b64 = encode_image(image_path)
    return [
        {
            "type": "text",
            "text": (
                f"Task: {task_prompt}\n\n"
                "Generate VQA questions for this robot manipulation video."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        },
    ]


def load_icl_examples(repo_root: Path) -> list:
    """Build few-shot user/assistant message pairs from the reference benchmark dataset.

    Returns a flat list of alternating user/assistant messages ready to be
    inserted into the messages list between the system prompt and the real request.
    """
    ref_dir = repo_root / "datasets" / "physical-ai-bench-generation"
    full_info_path = ref_dir / "cosmos_predict2_bench_full_info.json"

    if not full_info_path.exists():
        print("  [WARN] Reference full_info.json not found — skipping ICL examples")
        return []

    full_info = json.loads(full_info_path.read_text())
    prompt_map = {entry["video_id"]: entry["prompt_en"] for entry in full_info}

    messages = []
    for vid_id in ICL_EXAMPLE_IDS:
        image_path = ref_dir / "condition_image" / f"{vid_id}.jpg"
        vqa_path   = ref_dir / "vqa" / f"{vid_id}.json"

        if not image_path.exists() or not vqa_path.exists():
            print(f"  [WARN] ICL example {vid_id} missing image or vqa file — skipping")
            continue

        prompt   = prompt_map.get(vid_id, "")
        vqa_data = json.loads(vqa_path.read_text())

        # Convert benchmark VQA format → LLM output format (category + question + answer)
        llm_output = [
            {
                "category": entry["uid"].split("_(")[1].rstrip(")"),
                "question": entry["question"],
                "answer":   entry["answer"],
            }
            for entry in vqa_data
        ]

        messages.append({
            "role":    "user",
            "content": build_user_content(image_path, prompt),
        })
        messages.append({
            "role":    "assistant",
            "content": json.dumps(llm_output, indent=2),
        })

    return messages


def parse_llm_response(text: str) -> list:
    """Parse JSON array from LLM response, stripping any accidental markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def format_vqa_entries(raw: list, video_id: str) -> list:
    """Validate and convert LLM output into physical-ai-bench VQA format.

    Q numbers are assigned sequentially (1, 2, 3, ...) — they are just unique
    indices within the file; the category label in the uid carries the meaning.
    """
    entries = []
    for item in raw:
        category = item.get("category", "")
        question = item.get("question", "").strip()
        answer = str(item.get("answer", "")).upper()

        if category not in VALID_CATEGORIES:
            continue
        if answer not in ("A", "B"):
            continue
        if not question:
            continue

        entries.append({"category": category, "question": question, "answer": answer})

    # Assign sequential Q numbers after filtering
    result = []
    for idx, item in enumerate(entries, start=1):
        result.append({
            "question": item["question"],
            "index2ans": {"A": "yes", "B": "no"},
            "answer": item["answer"],
            "uid": f"{video_id}_Q{idx}_({item['category']})",
            "split": "val",
            "task": "task:success:discrete:True",
        })
    return result


def generate_vqa_for_video(
    video_id: str,
    task_prompt: str,
    frame_path: Path,
    model: str,
    icl_messages: list = (),
) -> tuple[list, float]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *icl_messages,
        {"role": "user", "content": build_user_content(frame_path, task_prompt)},
    ]
    response = completion(
        model=model,
        messages=messages,
        temperature=0.3,
        base_url=os.environ.get("LITELLM_BASE_URL"),
        api_key=os.environ.get("LITELLM_API_KEY")
    )
    # cost = completion_cost(completion_response=response)
    print('Raw response:', response.choices[0].message.content)
    raw = parse_llm_response(response.choices[0].message.content)

    entries = format_vqa_entries(raw, video_id)
    return entries, 0 # FIXME: change to completion cost if available


def main():
    repo_root = Path(__file__).parent.parent

    parser = argparse.ArgumentParser(
        description="Generate VQA questions for cosmos synthetic videos via LiteLLM"
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="Cosmos subset name (e.g. cosmos_predict2_2b_gr1_behavior)",
    )
    parser.add_argument(
        "--cosmos_data_dir",
        type=Path,
        default=repo_root / "cosmos_synthetic_data",
        help="Root dir of cosmos synthetic data (default: repo/cosmos_synthetic_data)",
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=repo_root / "datasets" / "cosmos_synthetic_data",
        help="Dataset dir with full_info.json (default: repo/datasets/cosmos_synthetic_data)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LiteLLM model string (default: gpt-4o)",
    )
    parser.add_argument(
        "--n_videos",
        type=int,
        default=None,
        help="Process only the first N videos (for testing)",
    )
    args = parser.parse_args()

    subset_dir = args.cosmos_data_dir / args.subset
    full_info_path = args.dataset_dir / args.subset / "full_info.json"
    vqa_dir = args.dataset_dir / args.subset / "vqa"

    if not subset_dir.exists():
        print(f"ERROR: subset directory not found: {subset_dir}")
        return
    if not full_info_path.exists():
        print(f"ERROR: full_info.json not found at {full_info_path}")
        print("       Run prepare_cosmos_synthetic_dataset.py --subset {args.subset} first.")
        return

    vqa_dir.mkdir(parents=True, exist_ok=True)

    with open(full_info_path) as f:
        full_info = json.load(f)

    if args.n_videos:
        full_info = full_info[: args.n_videos]

    icl_messages = load_icl_examples(repo_root)
    print(f"Subset     : {args.subset}")
    print(f"Model      : {args.model}")
    print(f"Videos     : {len(full_info)}")
    print(f"ICL shots  : {len(icl_messages) // 2}")
    print(f"Output dir : {vqa_dir}")
    print()

    total_cost = 0.0
    success = 0
    skipped = 0
    failed = 0

    for entry in full_info:
        video_id = entry["video_id"]
        prompt = entry["prompt_en"]
        # task_name is the original directory name without robot_ prefix
        task_name = entry.get("task_name", video_id.removeprefix("robot_"))
        frame_path = subset_dir / task_name / "frame_000000.jpg"

        if not frame_path.exists():
            print(f"  [SKIP] {video_id}: frame_000000.jpg not found")
            skipped += 1
            continue

        out_path = vqa_dir / f"{video_id}.json"
        if out_path.exists():
            print(f"  [SKIP] {video_id}: output already exists")
            skipped += 1
            continue

        try:
            entries, cost = generate_vqa_for_video(video_id, f"{prompt}\nTask: {task_name}", frame_path, args.model, icl_messages)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entries, f, indent=4, ensure_ascii=False)
            total_cost += cost
            success += 1
            print(f"  [OK]   {video_id}: {len(entries)} questions  cost=${cost:.5f}")
        except Exception as e:
            print(f"  [ERR]  {video_id}: {e}")
            failed += 1

    print()
    print(f"Done. Success={success}  Skipped={skipped}  Failed={failed}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    main()
