# ---
# jupyter:
#   jupytext:
#     formats: py:percent
# ---

# %% [markdown]
# # NB3 — DPO Training (the main event)
#
# **Stack:** TRL `DPOTrainer` + `DPOConfig(beta=0.1, lr=5e-7)` from deck §5.2.
# Maps to deck §3 (DPO derivation), §3.4 (failure modes — read closely!), §5.2 (TRL impl).
#
# > **Mục tiêu:** train DPO adapter on top of NB1 SFT-mini. Plot reward curves
# > (cả `chosen_rewards` và `rejected_rewards`). Save adapter to `adapters/dpo/`.
# >
# > Đây là **the** notebook quan trọng nhất của lab — 25/100 pts đến từ đây.
# > Đặc biệt là: **plot cả 2 curve riêng biệt**, không chỉ reward gap (deck §3.4).

# %% [markdown]
# ## 0. Setup

# %%
import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
from pathlib import Path

COMPUTE_TIER = os.environ.get("COMPUTE_TIER", "T4").upper()

if COMPUTE_TIER == "T4":
    BASE_MODEL = "unsloth/Qwen2.5-3B-bnb-4bit"
    MAX_LEN = 512
    MAX_PROMPT_LEN = 256
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 8
else:
    BASE_MODEL = "unsloth/Qwen2.5-7B-bnb-4bit"
    MAX_LEN = 1024
    MAX_PROMPT_LEN = 512
    PER_DEVICE_BATCH = 1
    GRAD_ACCUM = 4

# Hyperparameters from deck §5.2 (TRL DPOTrainer implementation frame)
BETA = float(os.environ.get("DPO_BETA", "0.1"))
LR = float(os.environ.get("DPO_LR", "5e-7"))
EPOCHS = int(os.environ.get("DPO_EPOCHS", "1"))

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
SFT_PATH = REPO_ROOT / "adapters" / "sft-mini"
DPO_OUT = REPO_ROOT / "adapters" / "dpo"
PREF_PATH = REPO_ROOT / "data" / "pref" / "train.parquet"

DPO_OUT.mkdir(parents=True, exist_ok=True)

assert SFT_PATH.exists(), f"NB1 must run first — {SFT_PATH} missing"
assert PREF_PATH.exists(), f"NB2 must run first — {PREF_PATH} missing"

print(f"COMPUTE_TIER:    {COMPUTE_TIER}")
print(f"BASE_MODEL:      {BASE_MODEL}")
print(f"DPO hyperparams: beta={BETA}  lr={LR}  epochs={EPOCHS}")
print(f"max_length:      {MAX_LEN}  (prompt={MAX_PROMPT_LEN})")
print(f"effective batch: {PER_DEVICE_BATCH * GRAD_ACCUM}")
print(f"SFT input:       {SFT_PATH}")
print(f"output:          {DPO_OUT}")

# %%
import torch

assert torch.cuda.is_available(), "DPO needs a CUDA GPU. See HARDWARE-GUIDE.md."

# %% [markdown]
# ## 1. Load policy + reference (the VRAM story)
#
# **Critical:** DPO scores each answer under the policy (trainable) AND a frozen
# reference. With PEFT we do **not** load a second model -- TRL toggles the LoRA
# adapter *off* to get the reference forward pass on the same 4-bit base. The
# extra VRAM vs SFT comes from two forward passes + holding chosen AND rejected
# sequences, not from a second copy of the weights.

# %%
from unsloth import FastLanguageModel
from peft import PeftModel

# Force disable xFormers on T4 to avoid xformers GQA backward NotImplementedError
if COMPUTE_TIER == "T4":
    FastLanguageModel.disable_xFormers = True

# Policy — gets new DPO LoRA adapter on top of SFT LoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    max_seq_length=MAX_LEN,
    dtype=None,
    load_in_4bit=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

from unsloth import get_chat_template
tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
)

# Load SFT adapter on top of base
model = PeftModel.from_pretrained(model, str(SFT_PATH), is_trainable=True)
print(f"Policy: {model.__class__.__name__} with SFT adapter loaded")

# %%
# Wrap policy with NEW LoRA adapter for DPO updates (don't merge SFT — keep stacked)
# Unsloth re-applies LoRA on top of the existing PeftModel.
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
    use_rslora=False,
    loftq_config=None,
)
print(f"Trainable params (DPO LoRA): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# %% [markdown]
# > **Why no separate `ref_model=` argument?** Modern TRL (≥ 0.12) auto-detects
# > PEFT models and uses the *base model without the adapter* as the reference.
# > That's the same memory layout: 1 base + 2 adapter sets in VRAM. No deepcopy
# > needed.

# %% [markdown]
# ## 2. Build DPOConfig (deck §5.2 hyperparameters)

# %%
from trl import DPOConfig

dpo_config = DPOConfig(
    output_dir=str(DPO_OUT.parent / "dpo-checkpoints"),
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LR,
    beta=BETA,
    max_length=MAX_LEN,
    max_prompt_length=MAX_PROMPT_LEN,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="no",
    optim="adamw_8bit",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    seed=42,
    loss_type="sigmoid",         # DPO standard (alternatives: ipo, hinge, kto)
    report_to="none",
)

print(f"DPOConfig: beta={dpo_config.beta}  lr={dpo_config.learning_rate}  loss_type={dpo_config.loss_type}")

# %% [markdown]
# ## 3. Load preference data

# %%
from datasets import Dataset

pref_ds = Dataset.from_parquet(str(PREF_PATH))
print(f"Loaded {len(pref_ds)} preference pairs from {PREF_PATH}")
print(f"Columns: {pref_ds.column_names}")

# %% [markdown]
# ## 4. Train

# %%
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    ref_model=None,                # auto-derived from PEFT base
    args=dpo_config,
    train_dataset=pref_ds,
    processing_class=tokenizer,
)

# %%
train_result = trainer.train()
print(f"\nFinal DPO loss: {train_result.training_loss:.4f}")

# %% [markdown]
# ## 5. Plot reward curves — THE diagnostic
#
# **Read deck §3.4 before interpreting these.** A growing reward gap can come from:
# - **(intended)** chosen reward going up + rejected staying flat
# - **(intended)** chosen rising slowly + rejected falling fast
# - **(likelihood displacement)** chosen reward going *down* + rejected falling faster
#
# The third case is what Razin et al. 2024 documented. It's not a bug, but it
# tells you the model is finding a way to widen the gap that doesn't necessarily
# improve actual chosen probability.

# %%
import matplotlib.pyplot as plt
import pandas as pd

logs = pd.DataFrame(trainer.state.log_history)
logs = logs[logs["loss"].notna() if "loss" in logs.columns else logs.index].copy()

# TRL DPO logs include rewards/chosen, rewards/rejected, rewards/margins, kl
chosen_col = "rewards/chosen" if "rewards/chosen" in logs.columns else None
rejected_col = "rewards/rejected" if "rewards/rejected" in logs.columns else None

# ── Save raw reward history to CSV/JSON for reproducibility ──
eval_dir = REPO_ROOT / "data" / "eval"
eval_dir.mkdir(parents=True, exist_ok=True)

reward_cols = ["step", "loss"]
if chosen_col:
    reward_cols.append(chosen_col)
if rejected_col:
    reward_cols.append(rejected_col)
if "rewards/margins" in logs.columns:
    reward_cols.append("rewards/margins")

reward_history = logs[[c for c in reward_cols if c in logs.columns]].dropna(subset=["loss"]).copy()
if chosen_col and rejected_col:
    reward_history["reward_gap"] = reward_history[chosen_col] - reward_history[rejected_col]
reward_history.to_csv(eval_dir / "dpo_reward_history.csv", index=False)
reward_history.to_json(eval_dir / "dpo_reward_history.json", orient="records", indent=2)
print(f"Saved raw reward history ({len(reward_history)} rows) to data/eval/dpo_reward_history.csv")
print(f"Saved raw reward history ({len(reward_history)} rows) to data/eval/dpo_reward_history.json")

# ── Plot 3 curves: chosen reward, rejected reward, reward gap ──
fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))

if chosen_col and rejected_col:
    steps = logs["step"].values
    chosen_vals = logs[chosen_col].values
    rejected_vals = logs[rejected_col].values
    gap_vals = chosen_vals - rejected_vals

    # Smoothing helper (simple rolling mean, window=5)
    def smooth(arr, w=5):
        if len(arr) < w:
            return arr
        kernel = pd.Series(arr).rolling(w, min_periods=1, center=True).mean().values
        return kernel

    # Panel 1: Chosen reward
    axes[0].plot(steps, chosen_vals, alpha=0.35, color="#2e548a", linewidth=0.8, label="raw")
    axes[0].plot(steps, smooth(chosen_vals), color="#2e548a", linewidth=2.0, label="smoothed")
    axes[0].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Implicit reward (log π/π_ref)")
    axes[0].set_title("Chosen reward")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Rejected reward
    axes[1].plot(steps, rejected_vals, alpha=0.35, color="#c83538", linewidth=0.8, label="raw")
    axes[1].plot(steps, smooth(rejected_vals), color="#c83538", linewidth=2.0, label="smoothed")
    axes[1].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Implicit reward (log π/π_ref)")
    axes[1].set_title("Rejected reward")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Reward gap (chosen − rejected)
    axes[2].plot(steps, gap_vals, alpha=0.35, color="#1a3355", linewidth=0.8, label="raw")
    axes[2].plot(steps, smooth(gap_vals), color="#1a3355", linewidth=2.0, label="smoothed")
    axes[2].axhline(0, color="#888", linestyle=":", linewidth=0.7)
    axes[2].set_xlabel("Training step")
    axes[2].set_ylabel("Reward gap (chosen − rejected)")
    axes[2].set_title("Reward gap (the headline number)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
else:
    for ax in axes:
        ax.text(0.5, 0.5, "No reward columns in trainer.state.log_history.\nLikely TRL version mismatch.",
                ha="center", va="center", transform=ax.transAxes)

fig.suptitle(f"DPO reward curves · {COMPUTE_TIER} · β={BETA} · lr={LR}", y=1.02)
fig.tight_layout()

screenshot_dir = REPO_ROOT / "submission" / "screenshots"
screenshot_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(screenshot_dir / "03-dpo-reward-curves.png", dpi=120, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 5a. Failure-mode self-check
#
# Read this cell carefully — it tells you which kind of "reward gap up" you got.

# %%
if chosen_col and rejected_col and len(logs) >= 5:
    last_chosen = logs[chosen_col].iloc[-5:].mean()
    last_rejected = logs[rejected_col].iloc[-5:].mean()
    last_gap = last_chosen - last_rejected
    first_chosen = logs[chosen_col].iloc[:5].mean()

    chosen_delta = last_chosen - first_chosen

    print(f"END  chosen reward:    {last_chosen:+.3f}")
    print(f"END  rejected reward:  {last_rejected:+.3f}")
    print(f"END  reward gap:       {last_gap:+.3f}")
    print()

    if last_gap < 0:
        print("✗ FAILURE: reward gap went NEGATIVE. DPO did the opposite of what you wanted.")
        print("  Likely causes: data quality (chosen/rejected swapped?), beta too high, lr too low.")
    elif chosen_delta < -0.5 and last_gap > 0:
        print("⚠  LIKELIHOOD DISPLACEMENT (deck §3.4):")
        print(f"   Reward gap is positive ({last_gap:+.3f}) — good!")
        print(f"   But chosen reward FELL by {chosen_delta:+.3f} during training.")
        print("   The gap grew because rejected fell faster than chosen.")
        print("   Document this in REFLECTION § 3 — it's a teachable moment, not a bug.")
    elif chosen_delta > 0 and last_gap > 0:
        print("✓ INTENDED: chosen reward UP and gap positive. Classic DPO success.")
    else:
        print("?  AMBIGUOUS: weak chosen movement + positive gap. Try longer training or higher lr.")

# %% [markdown]
# ## 6. Save adapter

# %%
trainer.model.save_pretrained(str(DPO_OUT))
tokenizer.save_pretrained(str(DPO_OUT))
print(f"Saved DPO adapter to {DPO_OUT}")

# Save the headline metrics for verify.py + REFLECTION
import json

# Verify DPO adapter is distinct from SFT adapter
assert str(DPO_OUT) != str(SFT_PATH), "DPO adapter must save to a different directory than SFT!"

metrics = {
    "compute_tier": COMPUTE_TIER,
    "base_model": BASE_MODEL,
    "beta": BETA,
    "lr": LR,
    "epochs": EPOCHS,
    "final_train_loss": float(train_result.training_loss),
    "end_chosen_reward": float(last_chosen) if chosen_col else None,
    "end_rejected_reward": float(last_rejected) if rejected_col else None,
    "end_reward_gap": float(last_gap) if chosen_col and rejected_col else None,
    "chosen_delta": float(chosen_delta) if chosen_col else None,
    "failure_mode": (
        "likelihood_displacement" if chosen_col and chosen_delta < -0.5 and last_gap > 0
        else "classic_success" if chosen_col and chosen_delta > 0 and last_gap > 0
        else "negative_gap" if chosen_col and last_gap < 0
        else "ambiguous" if chosen_col
        else "unknown"
    ),
}
(DPO_OUT / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"Wrote metrics to {DPO_OUT / 'dpo_metrics.json'}")
print(f"Failure mode detected: {metrics['failure_mode']}")

# Push DPO adapter to HuggingFace Hub
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("Uploading DPO adapter to HuggingFace Hub...")
        trainer.model.push_to_hub("AnhNQ-2A202600608/2A202600608-Nguyen-Quang-Anh-Day22-DPO", token=hf_token)
        tokenizer.push_to_hub("AnhNQ-2A202600608/2A202600608-Nguyen-Quang-Anh-Day22-DPO", token=hf_token)
        print("✓ Successfully pushed DPO adapter to HuggingFace Hub!")
    else:
        print("WARN: HF_TOKEN not found in environment. Skipping HuggingFace upload.")
except Exception as e:
    print(f"WARN: Failed to push DPO adapter to HuggingFace Hub: {e}")

# %% [markdown]
# ## 7. Vibe-coding callout
#
# Now's the time for the **β experiment** if you want the +6 rigor add-on.
#
# `make beta-sweep` runs this notebook 3 times with `DPO_BETA ∈ {0.05, 0.1, 0.5}`
# and saves to `adapters/dpo-b{0.05,0.1,0.5}/`. Plot the results yourself:
#
# ```python
# import json
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# results = []
# for d in sorted((REPO_ROOT / "adapters").glob("dpo-b*")):
#     m = json.loads((d / "dpo_metrics.json").read_text())
#     results.append((m["beta"], m["end_reward_gap"]))
# # plot β vs reward_gap
# ```
#
# **Think-hard zone:** what's the *expected* shape of the β-vs-reward-gap curve?
# Hypothesize before you look at the data. (Hint: deck §3.3.)
#
# **Next:** NB4 — qualitative side-by-side comparison.
