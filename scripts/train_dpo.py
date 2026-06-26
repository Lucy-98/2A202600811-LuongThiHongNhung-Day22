#!/usr/bin/env python3
"""CLI wrapper for NB3 logic — trains a DPO adapter.

Usage:
    python scripts/train_dpo.py
    python scripts/train_dpo.py --beta 0.05 --output-dir adapters/dpo-b0.05
    python scripts/train_dpo.py --beta 0.5  --output-dir adapters/dpo-b0.50

Mirrors `notebooks/03_dpo_train.py`. Used by `make beta-sweep` for the rigor
add-on +6 (β-sweep mini-experiment).
"""
from __future__ import annotations

import argparse
import json
import os
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--sft-path", default=str(REPO / "adapters" / "sft-mini"))
    parser.add_argument("--pref-path", default=str(REPO / "data" / "pref" / "train.parquet"))
    parser.add_argument("--output-dir", default=str(REPO / "adapters" / "dpo"))
    args = parser.parse_args()

    tier = os.environ.get("COMPUTE_TIER", "T4").upper()
    if tier == "T4":
        base_model = "unsloth/Qwen2.5-3B-bnb-4bit"
        max_len, max_prompt = 512, 256
        batch, grad_accum = 1, 8
    else:
        base_model = "unsloth/Qwen2.5-7B-bnb-4bit"
        max_len, max_prompt = 1024, 512
        batch, grad_accum = 1, 4

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print(f"Tier:       {tier}")
    print(f"Base:       {base_model}")
    print(f"Beta / LR:  {args.beta} / {args.lr}")
    print(f"Output:     {output}")

    import torch
    from datasets import Dataset
    from peft import PeftModel
    from trl import DPOConfig, DPOTrainer
    from unsloth import FastLanguageModel

    # Force disable xFormers on T4 to avoid xformers GQA backward NotImplementedError
    if tier == "T4":
        FastLanguageModel.disable_xFormers = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model, max_seq_length=max_len, dtype=None, load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from unsloth import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml",
    )

    model = PeftModel.from_pretrained(model, args.sft_path, is_trainable=True)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=42, use_rslora=False, loftq_config=None,
    )

    config = DPOConfig(
        output_dir=str(output.parent / f"{output.name}-checkpoints"),
        per_device_train_batch_size=batch,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=max_len,
        max_prompt_length=max_prompt,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="no",
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=42,
        loss_type="sigmoid",
        report_to="none",
    )

    pref = Dataset.from_parquet(args.pref_path)
    trainer = DPOTrainer(
        model=model, ref_model=None, args=config,
        train_dataset=pref, processing_class=tokenizer,
    )
    train_result = trainer.train()

    trainer.model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))

    # Headline metrics + full reward history
    import pandas as pd

    logs = pd.DataFrame(trainer.state.log_history)
    chosen_col = "rewards/chosen" if "rewards/chosen" in logs.columns else None
    rejected_col = "rewards/rejected" if "rewards/rejected" in logs.columns else None

    # Save raw reward history to CSV
    reward_cols = ["step", "loss"]
    if chosen_col: reward_cols.append(chosen_col)
    if rejected_col: reward_cols.append(rejected_col)
    reward_history = logs[[c for c in reward_cols if c in logs.columns]].dropna(subset=["loss"]).copy()
    if chosen_col and rejected_col:
        reward_history["reward_gap"] = reward_history[chosen_col] - reward_history[rejected_col]
    reward_history.to_csv(output / "reward_history.csv", index=False)

    last_chosen = float(logs[chosen_col].iloc[-5:].mean()) if chosen_col else None
    last_rejected = float(logs[rejected_col].iloc[-5:].mean()) if rejected_col else None
    first_chosen = float(logs[chosen_col].iloc[:5].mean()) if chosen_col else None
    chosen_delta = (last_chosen - first_chosen) if last_chosen is not None and first_chosen is not None else None
    end_gap = (last_chosen - last_rejected) if last_chosen is not None and last_rejected is not None else None

    metrics = {
        "compute_tier": tier,
        "base_model": base_model,
        "beta": args.beta,
        "lr": args.lr,
        "epochs": args.epochs,
        "final_train_loss": float(train_result.training_loss),
        "end_chosen_reward": last_chosen,
        "end_rejected_reward": last_rejected,
        "end_reward_gap": end_gap,
        "chosen_delta": chosen_delta,
        "failure_mode": (
            "likelihood_displacement" if chosen_delta is not None and chosen_delta < -0.5 and end_gap is not None and end_gap > 0
            else "classic_success" if chosen_delta is not None and chosen_delta > 0 and end_gap is not None and end_gap > 0
            else "negative_gap" if end_gap is not None and end_gap < 0
            else "ambiguous" if chosen_col
            else "unknown"
        ),
    }

    (output / "dpo_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nFinal loss:     {train_result.training_loss:.4f}")
    if end_gap is not None:
        print(f"End reward gap: {end_gap:+.3f}")
    print(f"Failure mode:   {metrics['failure_mode']}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
