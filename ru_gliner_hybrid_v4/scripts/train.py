"""Actual fine-tuning entry point. No model/data downloads unless --allow-download."""

from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ru_pii.schema import read_jsonl, sha256_file, LABELS


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/train_small.json")
    p.add_argument(
        "--model", help="local GLiNER directory or explicitly permitted Hub model"
    )
    p.add_argument("--revision", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--allow-download", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="2 training steps / 48 train / 12 dev; NOT a quality run",
    )
    p.add_argument(
        "--external",
        action="append",
        default=[],
        help="additional audited canonical TRAIN JSONL; total capped at 25%% of base train",
    )
    p.add_argument(
        "--allow-slow-cpu",
        action="store_true",
        help="explicitly allow the full profile on CPU; memory/time are not guaranteed",
    )
    p.add_argument(
        "--prepare-only",
        action="store_true",
        help="load real model and check token alignment without optimisation",
    )
    p.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="skip automatic dev calibration + post-train benchmark",
    )
    p.add_argument(
        "--benchmark-external",
        choices=["auto", "required", "none"],
        default="auto",
        help="after full training: try Russian external benchmarks; auto continues if offline",
    )
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    model_id = args.model or cfg["model_id"]
    revision = args.revision or (
        cfg.get("revision") if model_id == cfg["model_id"] else None
    )
    out = Path(args.output or cfg["output_dir"])
    if args.smoke and args.output is None:
        out = Path(str(out) + "-smoke")
    if out.exists() and any(out.iterdir()):
        p.error(
            "output directory must be empty; do not overwrite a checkpoint/manifest"
        )
    out.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "started",
        "training_executed": False,
        "weights_exported": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "config": cfg,
        "model_id": model_id,
        "revision": revision,
        "smoke_only": args.smoke,
        "external_files": args.external,
    }

    def save_status():
        (out / "RUN_STATUS.json").write_text(
            json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    save_status()
    try:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("WANDB_DISABLED", "true")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if not args.allow_download:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            if not Path(model_id).is_dir():
                raise FileNotFoundError(
                    "local checkpoint missing; supply --model or explicitly --allow-download"
                )
        import torch
        from gliner import GLiNER
        from transformers import set_seed
        from ru_pii.inference import GLiNERBackend, RussianPIIDetector
        from ru_pii.prepare import prepare_records

        set_seed(cfg["seed"])
        device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else args.device
        )
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if (
            device == "cpu"
            and not args.smoke
            and not args.prepare_only
            and not args.allow_slow_cpu
        ):
            raise RuntimeError("full CPU profile requires explicit --allow-slow-cpu")
        status["device"] = device
        status["versions"] = {
            n: importlib.metadata.version(n)
            for n in ["gliner", "transformers", "torch", "huggingface-hub"]
        }
        train = read_jsonl(cfg["train_path"])
        dev = read_jsonl(cfg["dev_path"])
        if any(r.get("split") != "train" for r in train) or any(
            r.get("split") != "dev" for r in dev
        ):
            raise ValueError("base train/dev split labels disagree with their roles")

        def text_key(r):
            import hashlib

            return hashlib.sha256(r["text"].casefold().encode()).hexdigest()

        known = {text_key(r) for r in train}
        dev_keys = {text_key(r) for r in dev}
        if known & dev_keys:
            raise ValueError("train/dev text leakage")
        external = []
        for path in args.external:
            records = read_jsonl(path)
            for r in records:
                if (
                    r.get("split") != "train"
                    or "pii-bench" in r.get("source", "")
                    or "pii_benchmark" in r.get("source", "")
                ):
                    raise ValueError("held-out benchmark cannot enter training")
                k = text_key(r)
                if k in dev_keys:
                    raise ValueError("external corpus overlaps the dev set")
                if k not in known:
                    external.append(r)
                    known.add(k)
        random.Random(cfg["seed"]).shuffle(external)
        status["external_accepted"] = min(len(external), len(train) // 4)
        train += external[: len(train) // 4]
        if args.smoke:
            train, dev = train[:48], dev[:12]
        status["data_sha256"] = {
            path: sha256_file(path)
            for path in [cfg["train_path"], cfg["dev_path"], *args.external]
        }
        load = {
            "load_tokenizer": True,
            "local_files_only": not args.allow_download,
            "map_location": "cpu",
        }
        if revision:
            load["revision"] = revision
        model = GLiNER.from_pretrained(model_id, **load)
        # Stable explicit type inventory: never randomly turn unannotated classes into negatives.
        # Keep the checkpoint's architectural type limit. Preparation splits
        # records whose annotated label inventory is larger than this limit.
        model.config.random_drop = False
        model.config.shuffle_types = False

        # GLiNER 0.2.29 emits out-of-range padding span indices near the end of
        # short sequences. CPU tolerates these, but CUDA asserts in indexSelect.
        original_collator_factory = model._create_data_collator

        class SafeCollator:
            def __init__(self, delegate):
                self.delegate = delegate

            def __call__(self, batch, **call_kwargs):
                model_batch = self.delegate(batch, **call_kwargs)
                span_idx = model_batch.get("span_idx")
                span_mask = model_batch.get("span_mask")
                text_lengths = model_batch.get("text_lengths")
                if span_idx is not None and span_mask is not None and text_lengths is not None:
                    valid_ends = text_lengths.view(-1, 1).to(span_idx.device)
                    invalid = span_idx[..., 1] >= valid_ends
                    span_mask = span_mask & ~invalid
                    safe_idx = span_idx.clone()
                    safe_idx[..., 0].clamp_(min=0)
                    safe_idx[..., 1].clamp_(min=0)
                    safe_idx[..., 1] = torch.minimum(safe_idx[..., 1], valid_ends - 1)
                    safe_idx[..., 0] = torch.minimum(safe_idx[..., 0], safe_idx[..., 1])
                    model_batch["span_idx"] = safe_idx
                    model_batch["span_mask"] = span_mask
                return model_batch

        def safe_collator_factory(**kwargs):
            return SafeCollator(original_collator_factory(**kwargs))

        model._create_data_collator = safe_collator_factory
        backend = GLiNERBackend(
            model,
            subtoken_budget=cfg["subtoken_budget"],
            word_window=cfg["word_window"],
        )
        prepared_train, train_stats = prepare_records(
            train, backend, overlap=cfg["overlap"]
        )
        prepared_dev, dev_stats = prepare_records(dev, backend, overlap=cfg["overlap"])
        status["preparation"] = {
            "train": train_stats,
            "dev": dev_stats,
            "max_width": backend.max_width,
        }
        status["status"] = "prepared"
        save_status()
        if args.prepare_only:
            return
        model.to(device)
        # First real inference confirms this installed GLiNER build accepts the wrapper.
        RussianPIIDetector(backend, batch_size=1).predict(
            "Почта клиента: test@example.org"
        )
        bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
        fp16 = device == "cuda" and not bf16
        if cfg.get("mixed_precision") == "none":
            bf16 = fp16 = False
        steps = 2 if args.smoke else cfg["max_steps"]
        every = 1 if args.smoke else cfg["eval_steps"]
        status["status"] = "optimising"
        status["optimisation_started"] = True
        status["training_executed"] = (
            None  # unknown until trainer returns; not a claim of zero partial steps
        )
        save_status()
        # train_model performs trainer.train internally (official GLiNER API).
        trainer = model.train_model(
            train_dataset=prepared_train,
            eval_dataset=prepared_dev,
            output_dir=str(out),
            max_steps=steps,
            learning_rate=cfg["learning_rate"],
            others_lr=cfg["others_lr"],
            per_device_train_batch_size=1
            if args.smoke
            else cfg["per_device_train_batch_size"],
            per_device_eval_batch_size=1
            if args.smoke
            else cfg["per_device_eval_batch_size"],
            gradient_accumulation_steps=1
            if args.smoke
            else cfg["gradient_accumulation_steps"],
            weight_decay=cfg["weight_decay"],
            warmup_ratio=cfg["warmup_ratio"],
            lr_scheduler_type="cosine",
            max_grad_norm=cfg["max_grad_norm"],
            eval_strategy="steps",
            eval_steps=every,
            save_strategy="steps",
            save_steps=every,
            save_total_limit=2,
            logging_steps=1 if args.smoke else cfg["logging_steps"],
            logging_first_step=True,
            logging_strategy="steps",
            disable_tqdm=False,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=cfg["gradient_checkpointing"],
            dataloader_num_workers=0,
            report_to="none",
            seed=cfg["seed"],
            data_seed=cfg["seed"],
        )
        status["training_executed"] = True
        status["global_step"] = int(trainer.state.global_step)
        status["training_log"] = trainer.state.log_history
        training_log = trainer.state.log_history
        (out / "training_log.json").write_text(
            json.dumps(training_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (out / "training_log.jsonl").open("w", encoding="utf-8") as log_file:
            for event in training_log:
                log_file.write(json.dumps(event, ensure_ascii=False) + "\n")
        print("Training loss history:")
        for event in training_log:
            metrics = []
            for key in ("loss", "eval_loss", "learning_rate"):
                if key in event:
                    metrics.append(f"{key}={event[key]:.6g}")
            if metrics and "step" in event:
                print(f"  step={int(event['step'])} " + " ".join(metrics), flush=True)
        trainer.save_model(str(out))
        # Include resolved backbone config and tokenizers, not just a weights file.
        model.save_pretrained(str(out), safe_serialization=True)
        (out / "labels.json").write_text(
            json.dumps(LABELS, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        status["weights_exported"] = True
        status["quality_claim"] = None

        # A smoke run proves integration only; do not spend time or accidentally
        # publish quality numbers from two optimisation steps. Full training, on
        # the other hand, automatically calibrates thresholds on DEV and then
        # evaluates held-out local sets + Russian external benchmarks.
        if args.smoke or args.skip_benchmark:
            status["status"] = (
                "smoke_completed"
                if args.smoke
                else "fine_tuning_completed_benchmark_skipped"
            )
        else:
            from ru_pii.metrics import calibrate
            from scripts.post_train_benchmark import run_suite

            status["status"] = "calibrating_on_dev"
            save_status()
            # Collect candidates at a deliberately low threshold; calibration
            # itself selects per-label thresholds using DEV only.
            low_thresholds = {k: 0.05 for k in LABELS}
            calibration_detector = RussianPIIDetector(
                GLiNERBackend(
                    model,
                    subtoken_budget=cfg["subtoken_budget"],
                    word_window=cfg["word_window"],
                ),
                thresholds=low_thresholds,
                batch_size=cfg["per_device_eval_batch_size"],
                overlap=cfg["overlap"],
            )
            dev_predictions = []
            dev_texts = [r["text"] for r in dev]
            bs = max(1, cfg["per_device_eval_batch_size"])
            for i in range(0, len(dev_texts), bs):
                batch = calibration_detector.predict_batch(
                    dev_texts[i : i + bs], include_auxiliary=True
                )
                dev_predictions.extend([[e.to_dict() for e in doc] for doc in batch])
            selected = calibrate(dev, dev_predictions, beta=2.0, min_support=10)
            (out / "thresholds.json").write_text(
                json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            status["thresholds_saved"] = True
            status["threshold_selection_split"] = "dev"
            status["status"] = "benchmarking"
            save_status()
            # Release the training model before loading the exported checkpoint for benchmarking.
            del calibration_detector
            del backend
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            bench_path = out / "benchmark.json"
            suite = run_suite(
                model=str(out),
                device=device,
                output=str(bench_path),
                batch_size=max(1, cfg["per_device_eval_batch_size"]),
                external=args.benchmark_external,
            )
            status["benchmark_file"] = str(bench_path)
            status["benchmarks"] = {
                k: v.get("status") for k, v in suite["benchmarks"].items()
            }
            status["status"] = "fine_tuning_and_benchmark_completed"

        status["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_status()
    except Exception as exc:
        status["status"] = "failed"
        status["failure_type"] = type(exc).__name__
        # Deliberately omit exception values/traceback locals: may contain data.
        save_status()
        print(
            f"Run failed ({type(exc).__name__}); inspect local configuration and RUN_STATUS.json.",
            file=sys.stderr,
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
