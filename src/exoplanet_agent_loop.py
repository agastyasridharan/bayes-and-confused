"""
Stage 2 (Exoplanet domain): Trajectory Collection + Activation Extraction

Mirrors agent_loop.py for the exoplanet domain. The tool exchange is simulated
exactly as in MP — empty result is `{"results": []}`, data-present result is
`{"results": [{"pl_name": ..., <prop>: <value>}]}` — so the activation
distribution differs only in tool name, parameter names, and field schema.

Empty side:   200 perturbations × 5 system prompts = 1000 trajectories.
Data-present: 200 real planets   × 5 system prompts = 1000 trajectories.
Total: 2000.

Activations saved as per-trajectory .npy files of shape [n_layers+1, hidden_dim].

Usage:
  python src/exoplanet_agent_loop.py
  python src/exoplanet_agent_loop.py --smoke-test
"""

import argparse
import json
import random
import sys
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

# Reuse the domain-agnostic forward+generate helper
sys.path.insert(0, str(ROOT / "src"))
from agent_loop import extract_and_generate

# ---------------------------------------------------------------------------
# Templates and system prompts (exoplanet variants)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS: dict[str, str] = {}
_current_key = None
with open(ROOT / "prompts" / "exoplanet_system_prompts.md") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("## "):
            _current_key = line[3:].strip()
            SYSTEM_PROMPTS[_current_key] = []
        elif _current_key and line and not line.startswith("#"):
            SYSTEM_PROMPTS[_current_key].append(line)
SYSTEM_PROMPTS = {k: " ".join(v) for k, v in SYSTEM_PROMPTS.items()}

TEMPLATES: dict[str, list[str]] = {}
_current_prop = None
with open(ROOT / "prompts" / "exoplanet_templates.md") as f:
    for line in f:
        line = line.rstrip()
        if line.startswith("## "):
            _current_prop = line[3:].strip()
            TEMPLATES[_current_prop] = []
        elif _current_prop and (
            line.startswith('"')
            or (line and line[0].isdigit() and '"' in line)
        ):
            start = line.index('"') + 1
            end = line.rindex('"')
            TEMPLATES[_current_prop].append(line[start:end])

assert all(len(v) == 5 for v in TEMPLATES.values()), (
    f"Expected 5 templates per property, got: "
    f"{ {k: len(v) for k, v in TEMPLATES.items()} }"
)

PROPERTIES = list(TEMPLATES.keys())  # pl_orbper, pl_rade, pl_bmasse, pl_eqt

TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "lookup_exoplanet_property",
        "description": (
            "Look up a property of a confirmed exoplanet in the NASA Exoplanet "
            "Archive. Returns a list of matching entries, or an empty list if "
            "the planet is not found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pl_name": {
                    "type": "string",
                    "description": "Name of the exoplanet (e.g., 'Kepler-186 f')",
                },
                "property": {
                    "type": "string",
                    "enum": PROPERTIES,
                    "description": "The property to look up",
                },
            },
            "required": ["pl_name", "property"],
        },
    },
}


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_messages(
    system_prompt: str,
    user_query: str,
    pl_name: str,
    prop: str,
    tool_result_content: str,
) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup_exoplanet_property",
                        "arguments": json.dumps(
                            {"pl_name": pl_name, "property": prop}
                        ),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_result_content},
    ]


def make_empty_tool_result() -> str:
    return json.dumps({"results": []})


def make_data_present_tool_result(planet: dict, prop: str) -> str:
    return json.dumps({
        "results": [
            {
                "pl_name": planet["pl_name"],
                prop: planet[prop],
            }
        ]
    })


# ---------------------------------------------------------------------------
# Trajectory assignment
# ---------------------------------------------------------------------------

def assign_property_template(idx: int) -> tuple[str, int, str]:
    """Deterministically rotate through (property, template_index) pairs.
    With 4 properties × 5 templates = 20 cells, every 20 trajectories the
    full cycle repeats."""
    combos = [(p, t_idx) for p in PROPERTIES for t_idx in range(5)]
    p, t_idx = combos[idx % len(combos)]
    return p, t_idx, TEMPLATES[p][t_idx]


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectory(
    model, tokenizer, trajectory_id: str, side: str,
    pl_name: str, prop: str, tmpl_idx: int, user_query: str,
    system_prompt_variant: str, system_prompt: str,
    tool_result: str, act_dir: Path,
    extra_fields: dict | None = None,
) -> dict:
    messages = build_messages(system_prompt, user_query, pl_name, prop, tool_result)
    response, activations, last_tok, prompt_text = extract_and_generate(
        model, tokenizer, messages, tool_def=TOOL_DEF
    )
    act_path = act_dir / f"{trajectory_id}.npy"
    np.save(act_path, activations)

    record = {
        "trajectory_id": trajectory_id,
        "side": side,
        "pl_name": pl_name,
        "property": prop,
        "template_index": tmpl_idx,
        "system_prompt_variant": system_prompt_variant,
        "system_prompt": system_prompt,
        "user_query": user_query,
        "tool_result": tool_result,
        "assistant_response": response,
        "activation_path": str(act_path.relative_to(ROOT)),
        "activation_shape": list(activations.shape),
        "extraction_token": last_tok,
        "messages": messages,
    }
    if extra_fields:
        record.update(extra_fields)
    return record


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def run_smoke_test(model, tokenizer, perturbations, real_planets, act_dir):
    print("\n" + "=" * 60)
    print("SMOKE TEST: 5 trajectories")
    print("=" * 60)

    sp_keys = list(SYSTEM_PROMPTS.keys())
    test_specs = [
        ("empty",        0, sp_keys[0]),
        ("empty",        1, sp_keys[1]),
        ("empty",        2, sp_keys[2]),
        ("data_present", 0, sp_keys[0]),
        ("data_present", 1, sp_keys[1]),
    ]

    for i, (side, data_idx, sp_key) in enumerate(test_specs):
        prop, tmpl_idx, tmpl_str = assign_property_template(i)

        if side == "empty":
            pert = perturbations[data_idx]
            name = pert["perturbed_name"]
            tool_result = make_empty_tool_result()
        else:
            planet = real_planets[data_idx]
            name = planet["pl_name"]
            tool_result = make_data_present_tool_result(planet, prop)

        user_query = tmpl_str.format(pl_name=name)
        system_prompt = SYSTEM_PROMPTS[sp_key]

        tid = f"smoke_exo_{i}"
        messages = build_messages(system_prompt, user_query, name, prop, tool_result)
        response, activations, last_tok, prompt_text = extract_and_generate(
            model, tokenizer, messages, tool_def=TOOL_DEF
        )
        act_path = act_dir / f"{tid}.npy"
        np.save(act_path, activations)

        print(f"\n{'─'*60}")
        print(f"TRAJECTORY {i}: {side} / {sp_key} / {name} / {prop}")
        print(f"{'─'*60}")
        print(f"\n[FULL PROMPT]\n{prompt_text}")
        print(f"\n[EXTRACTION TOKEN] '{last_tok}'")
        print(f"[ACTIVATION SHAPE] {activations.shape}")
        print(f"\n[TOOL RESULT] {tool_result}")
        print(f"\n[RESPONSE]\n{response}")
        print()

    for i in range(5):
        act = np.load(act_dir / f"smoke_exo_{i}.npy")
        assert act.shape[0] == CFG["probe"]["n_layers"], (
            f"Expected {CFG['probe']['n_layers']} layers, got {act.shape[0]}"
        )

    print("=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_full(model, tokenizer, perturbations, real_planets, act_dir, out_dir):
    sp_keys = list(SYSTEM_PROMPTS.keys())
    all_trajectories = []
    t0 = time.time()

    n_empty_target = len(perturbations) * len(sp_keys)
    print(f"\n--- Empty-side trajectories (target: {n_empty_target}) ---")

    for pert_idx, pert in enumerate(perturbations):
        name = pert["perturbed_name"]
        for sp_idx, sp_key in enumerate(sp_keys):
            traj_idx = pert_idx * len(sp_keys) + sp_idx
            prop, tmpl_idx, tmpl_str = assign_property_template(traj_idx)
            user_query = tmpl_str.format(pl_name=name)

            rec = collect_trajectory(
                model, tokenizer,
                trajectory_id=f"exo_empty_{traj_idx:04d}",
                side="empty",
                pl_name=name,
                prop=prop, tmpl_idx=tmpl_idx,
                user_query=user_query,
                system_prompt_variant=sp_key,
                system_prompt=SYSTEM_PROMPTS[sp_key],
                tool_result=make_empty_tool_result(),
                act_dir=act_dir,
                extra_fields={
                    "source_hostname": pert["source_hostname"],
                    "next_letter": pert["next_letter"],
                    "system_size": pert["system_size"],
                    "existing_letters": pert["existing_letters"],
                },
            )
            all_trajectories.append(rec)

            if (traj_idx + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (traj_idx + 1) / elapsed
                eta = (n_empty_target - traj_idx - 1) / rate
                print(f"  {traj_idx+1}/{n_empty_target}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    print(f"  Collected {n_empty_target} empty-side trajectories.")

    n_present_target = len(real_planets) * len(sp_keys)
    print(f"\n--- Data-present trajectories (target: {n_present_target}) ---")
    t1 = time.time()

    for plan_idx, planet in enumerate(real_planets):
        name = planet["pl_name"]
        for sp_idx, sp_key in enumerate(sp_keys):
            traj_idx = plan_idx * len(sp_keys) + sp_idx
            prop, tmpl_idx, tmpl_str = assign_property_template(traj_idx)
            user_query = tmpl_str.format(pl_name=name)

            rec = collect_trajectory(
                model, tokenizer,
                trajectory_id=f"exo_present_{traj_idx:04d}",
                side="data_present",
                pl_name=name,
                prop=prop, tmpl_idx=tmpl_idx,
                user_query=user_query,
                system_prompt_variant=sp_key,
                system_prompt=SYSTEM_PROMPTS[sp_key],
                tool_result=make_data_present_tool_result(planet, prop),
                act_dir=act_dir,
                extra_fields={
                    "hostname": planet["hostname"],
                    "sy_pnum": planet["sy_pnum"],
                },
            )
            all_trajectories.append(rec)

            if (traj_idx + 1) % 50 == 0:
                elapsed = time.time() - t1
                rate = (traj_idx + 1) / elapsed
                eta = (n_present_target - traj_idx - 1) / rate
                print(f"  {traj_idx+1}/{n_present_target}  "
                      f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    print(f"  Collected {n_present_target} data-present trajectories.")

    out_path = out_dir / "exoplanet_trajectories.json"
    with open(out_path, "w") as f:
        json.dump(all_trajectories, f, indent=2)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Total: {len(all_trajectories)} trajectories in {total_time:.0f}s")
    print(f"Saved to {out_path}")

    for sp_key in sp_keys:
        subset = [t for t in all_trajectories
                  if t["side"] == "empty" and t["system_prompt_variant"] == sp_key]
        lens = [len(t["assistant_response"].split()) for t in subset]
        print(f"  empty/{sp_key}: n={len(subset)}, mean_words={np.mean(lens):.0f}")

    for sp_key in sp_keys:
        subset = [t for t in all_trajectories
                  if t["side"] == "data_present" and t["system_prompt_variant"] == sp_key]
        lens = [len(t["assistant_response"].split()) for t in subset]
        print(f"  present/{sp_key}: n={len(subset)}, mean_words={np.mean(lens):.0f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 5-trajectory diagnostic instead of full run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap perturbations and real planets at this many each "
                             "(yields N × n_prompts trajectories per side). "
                             "Default uses all in data/exoplanets.json.")
    args = parser.parse_args()

    print("Stage 2 (Exoplanet): Trajectory Collection + Activation Extraction")
    print("=" * 60)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Precision: {CFG['model']['precision']}")
        if not torch.cuda.is_bf16_supported():
            print("  WARNING: bf16 not supported on this GPU")
    elif torch.backends.mps.is_available():
        device = "mps"
        print("  Device: MPS (Apple Silicon)")
        print("  NOTE: spec requires A100+bf16; MPS acceptable for smoke test")
    else:
        device = "cpu"
        print("  WARNING: No GPU detected, running on CPU (very slow)")

    print(f"\nLoading {MODEL_NAME} (revision {MODEL_REVISION[:12]}...)...")
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

    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"  Loaded: {n_params/1e9:.1f}B params, {n_layers} layers, "
          f"hidden_dim={hidden_dim}")

    with open(ROOT / "data" / "exoplanets.json") as f:
        data = json.load(f)
    real_planets = data["real_planets"]
    perturbations = data["perturbations"]
    print(f"  {len(perturbations)} perturbations, {len(real_planets)} real planets")

    if args.limit is not None and not args.smoke_test:
        perturbations = perturbations[:args.limit]
        real_planets = real_planets[:args.limit]
        n_total = args.limit * len(SYSTEM_PROMPTS) * 2
        print(f"  --limit {args.limit}: capped to {len(perturbations)} perturbations "
              f"+ {len(real_planets)} real planets ({n_total} total trajectories)")

    out_dir = ROOT / "data" / "trajectories"
    act_dir = out_dir / "activations"
    out_dir.mkdir(parents=True, exist_ok=True)
    act_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_test:
        run_smoke_test(model, tokenizer, perturbations, real_planets, act_dir)
    else:
        run_full(model, tokenizer, perturbations, real_planets, act_dir, out_dir)


if __name__ == "__main__":
    main()
