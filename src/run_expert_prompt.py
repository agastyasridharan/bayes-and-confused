"""
Run the new 'expert' system prompt variant on all 200 perturbations + 200 real
materials. Appends to existing trajectory file.

Usage:
  python src/run_expert_prompt.py               # full 400 trajectories
  python src/run_expert_prompt.py --smoke-test   # 5 trajectories
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

random.seed(CFG["seeds"]["random"])
np.random.seed(CFG["seeds"]["numpy"])
torch.manual_seed(CFG["seeds"]["torch"])

MODEL_NAME = CFG["model"]["name"]
MODEL_REVISION = CFG["model"]["revision"]
_PRECISION_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
PRECISION = _PRECISION_MAP[CFG["model"]["precision"]]

# Re-use agent_loop infrastructure
import sys
sys.path.insert(0, str(ROOT / "src"))
from agent_loop import (
    TEMPLATES, PROPERTIES, TOOL_DEF,
    build_messages, make_empty_tool_result, make_data_present_tool_result,
    assign_property_template, extract_and_generate,
)

# Load the new expert prompt
SYSTEM_PROMPTS = {}
_current_key = None
with open(ROOT / "prompts" / "system_prompts.md") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("## "):
            _current_key = line[3:].strip()
            SYSTEM_PROMPTS[_current_key] = []
        elif _current_key and line and not line.startswith("#"):
            SYSTEM_PROMPTS[_current_key].append(line)
SYSTEM_PROMPTS = {k: " ".join(v) for k, v in SYSTEM_PROMPTS.items()}

EXPERT_PROMPT = SYSTEM_PROMPTS["expert"]
print(f"Expert prompt: {EXPERT_PROMPT[:80]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    print("Running 'expert' system prompt variant")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  Device: MPS (Apple Silicon)")
    else:
        device = "cpu"

    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION,
            torch_dtype=PRECISION, device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, revision=MODEL_REVISION,
            torch_dtype=PRECISION, low_cpu_mem_usage=True,
        )
        if device == "mps":
            model = model.to("mps")
    model.eval()
    print("  Model loaded.")

    # Load data
    with open(ROOT / "data" / "perturbations_verified.json") as f:
        perturbations = json.load(f)
    with open(ROOT / "data" / "materials.json") as f:
        data = json.load(f)
    real_materials = data["real_materials"]

    out_dir = ROOT / "data" / "trajectories"
    act_dir = out_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        perturbations = perturbations[:3]
        real_materials = real_materials[:2]

    new_trajectories = []
    t0 = time.time()

    # Empty side: 200 perturbations × expert prompt
    print(f"\n--- Empty-side (expert): {len(perturbations)} trajectories ---")
    for idx, pert in enumerate(perturbations):
        formula = pert["perturbed_formula"]
        prop, tmpl_idx, tmpl_str = assign_property_template(idx)
        user_query = tmpl_str.format(formula=formula)
        tool_result = make_empty_tool_result()

        messages = build_messages(EXPERT_PROMPT, user_query, formula, prop, tool_result)
        response, activations, last_tok, prompt_text = extract_and_generate(
            model, tokenizer, messages
        )

        tid = f"empty_expert_{idx:04d}"
        act_path = act_dir / f"{tid}.npy"
        np.save(act_path, activations)

        rec = {
            "trajectory_id": tid,
            "side": "empty",
            "formula": formula,
            "source_formula": pert["source_formula"],
            "source_material_id": pert.get("source_material_id", ""),
            "perturbation_type": pert["perturbation_type"],
            "property": prop,
            "template_index": tmpl_idx,
            "system_prompt_variant": "expert",
            "system_prompt": EXPERT_PROMPT,
            "user_query": user_query,
            "tool_result": tool_result,
            "assistant_response": response,
            "activation_path": str(act_path.relative_to(ROOT)),
            "activation_shape": list(activations.shape),
            "extraction_token": last_tok,
            "messages": messages,
        }
        new_trajectories.append(rec)

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1}/{len(perturbations)} ({elapsed:.0f}s)")

    # Data-present side: 200 real materials × expert prompt
    print(f"\n--- Data-present (expert): {len(real_materials)} trajectories ---")
    t1 = time.time()
    for idx, mat in enumerate(real_materials):
        formula = mat["formula"]
        prop, tmpl_idx, tmpl_str = assign_property_template(idx)
        user_query = tmpl_str.format(formula=formula)
        tool_result = make_data_present_tool_result(mat, prop)

        messages = build_messages(EXPERT_PROMPT, user_query, formula, prop, tool_result)
        response, activations, last_tok, prompt_text = extract_and_generate(
            model, tokenizer, messages
        )

        tid = f"present_expert_{idx:04d}"
        act_path = act_dir / f"{tid}.npy"
        np.save(act_path, activations)

        rec = {
            "trajectory_id": tid,
            "side": "data_present",
            "formula": formula,
            "material_id": mat["material_id"],
            "property": prop,
            "template_index": tmpl_idx,
            "system_prompt_variant": "expert",
            "system_prompt": EXPERT_PROMPT,
            "user_query": user_query,
            "tool_result": tool_result,
            "assistant_response": response,
            "activation_path": str(act_path.relative_to(ROOT)),
            "activation_shape": list(activations.shape),
            "extraction_token": last_tok,
            "messages": messages,
        }
        new_trajectories.append(rec)

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - t1
            print(f"  {idx+1}/{len(real_materials)} ({elapsed:.0f}s)")

    # Save: append to existing or write new file
    expert_path = out_dir / "expert_trajectories.json"
    with open(expert_path, "w") as f:
        json.dump(new_trajectories, f, indent=2)
    print(f"\nSaved {len(new_trajectories)} trajectories to {expert_path}")

    # Quick fabrication analysis
    import re
    empty_expert = [t for t in new_trajectories if t["side"] == "empty"]
    fab_broad = [t for t in empty_expert if "Based on" in t["assistant_response"]
                 or re.search(r'\d+\.?\d*\s*eV', t["assistant_response"])
                 or re.search(r'(?:approximately|around|roughly|about|estimated?)\s*\d', t["assistant_response"], re.I)]
    print(f"\nEmpty-side fabrication (heuristic): {len(fab_broad)}/{len(empty_expert)} "
          f"({len(fab_broad)/max(len(empty_expert),1):.1%})")

    # Show 10 random responses
    print(f"\n--- 10 random empty-side responses ---")
    for t in random.sample(empty_expert, min(10, len(empty_expert))):
        print(f"\n[{t['trajectory_id']}] {t['formula']} | {t['property']} | tmpl={t['template_index']}")
        print(f"  {t['assistant_response'][:250]}")


if __name__ == "__main__":
    main()
