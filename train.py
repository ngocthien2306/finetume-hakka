"""
Hakka <-> Mandarin Translation Fine-tuning
Usage:
  # Single GPU (default gpu 0)
  python train.py

  # Single GPU, pick device
  python train.py --gpu 1

  # Multi-GPU via torchrun (set CUDA_VISIBLE_DEVICES externally)
  CUDA_VISIBLE_DEVICES=0,4 torchrun --nproc_per_node=2 train.py

  # Custom paths / hyperparams
  python train.py --dataset-dir ./dataset --output-dir ./outputs --epochs 3 --lr 2e-4
"""

import argparse, csv, glob, json, math, os, random, re
from collections import Counter

# ── HuggingFace cache (module-level fallback) ─────────────────────────────────
# Overridden in main() via --hf-cache arg; this just ensures a safe default
# in case something imports HF at module level.
_hf_cache_default = "/root/nguyen/research/finetune_llm/hf_cache"
os.makedirs(_hf_cache_default, exist_ok=True)
os.environ.setdefault("HF_HOME", _hf_cache_default)


# ── Args ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu",          default="0",     help="GPU id(s), comma-separated, e.g. 0 or 0,1")
    p.add_argument("--dataset-dir",  default="./dataset")
    p.add_argument("--output-dir",   default="./outputs/hakka_translation")
    p.add_argument("--model",        default="yentinglin/Llama-3-Taiwan-8B-Instruct")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--batch",        type=int,   default=2)
    p.add_argument("--grad-accum",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=2e-4)
    p.add_argument("--max-len",      type=int,   default=2048)
    p.add_argument("--lora-r",       type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=16)
    p.add_argument("--train-ratio",  type=float, default=0.90)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--no-lexicon",   action="store_true", help="Skip HAT Lexicon word pairs")
    p.add_argument("--eval-only",    action="store_true", help="Skip training, run eval only")
    p.add_argument("--adapter-path", default=None, help="Load existing adapter for eval-only mode")
    p.add_argument("--hf-cache",     default="/root/nguyen/research/finetune_llm/hf_cache",
                                     help="HuggingFace model cache directory")
    return p.parse_args()


# ── System prompts ────────────────────────────────────────────────────────────

SYSTEM = {
    "h2m": (
        "你是專業翻譯助手。請將使用者輸入的客語漢字翻成自然、流暢、忠實原意的繁體中文。"
        "只輸出翻譯結果，不要加解釋、不要加前後綴。"
    ),
    "m2h_sixian": (
        "你是專業翻譯助手。請將使用者輸入的繁體中文翻成客語漢字（四縣腔）。"
        "只輸出翻譯結果，不要加解釋、不要加前後綴。"
    ),
    "m2h_hailu": (
        "你是專業翻譯助手。請將使用者輸入的繁體中文翻成客語漢字（海陸腔）。"
        "只輸出翻譯結果，不要加解釋、不要加前後綴。"
    ),
    "dialogue": (
        "你是一位精通客家語（四縣腔）的醫師。"
        "請使用客家語與病人進行問診，詢問必要的症狀細節。"
    ),
}

_FFF9 = "\ufff9"
_FFFB = "\ufffb"
MIN_LEN = 4


# ── Data helpers ──────────────────────────────────────────────────────────────

def _make(sys_key, human, gpt, hakka, mandarin, direction, source):
    return {
        "system": SYSTEM[sys_key],
        "conversations": [
            {"from": "human", "value": human},
            {"from": "gpt",   "value": gpt},
        ],
        "hakka": hakka, "mandarin": mandarin,
        "direction": direction, "source": source,
    }


def _clean(text):
    if not text or len(text) < MIN_LEN:
        return False
    latin = sum(1 for c in text if c.isascii() and (c.isalpha() or c.isdigit()))
    return (latin / max(len(text), 1)) < 0.25


def extract_tsv(data_dir):
    pairs, dialogues = [], []
    for fp in sorted(glob.glob(os.path.join(data_dir, "HAKKA_UTF8_TSV", "*.txt"))):
        with open(fp, "r", encoding="utf-16", errors="replace") as f:
            lines = [l.strip() for l in f if l.strip()]
        rows = []
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) >= 6:
                rows.append({"speaker": cols[1].strip(),
                             "hakka": cols[4].strip(), "mandarin": cols[5].strip()})
        for r in rows:
            h, m = r["hakka"], r["mandarin"]
            if not _clean(h) or not _clean(m):
                continue
            pairs.append(_make("h2m",        h, m, h, m, "h2m", "tsv"))
            pairs.append(_make("m2h_sixian", m, h, h, m, "m2h", "tsv"))
        history = []
        for r in rows:
            t = r["hakka"]
            if not t:
                continue
            if r["speaker"] == "speaker02":
                history.append({"from": "human", "value": t})
            elif r["speaker"] == "speaker01" and history:
                dialogues.append({
                    "system": SYSTEM["dialogue"],
                    "conversations": history.copy() + [{"from": "gpt", "value": t}],
                    "direction": "dialogue", "source": "tsv",
                })
                history.append({"from": "gpt", "value": t})
    print(f"  TSV:  {len(pairs)//2} sentence pairs | {len(dialogues)} dialogue examples")
    return pairs, dialogues


def extract_dict(data_dir):
    pairs, skipped, seen = [], Counter(), set()
    path = os.path.join(data_dir, "dict-hakka.json")
    if not os.path.exists(path):
        print("  DICT: not found, skipping"); return pairs
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for entry in data:
        title = entry.get("title", "")
        for het in entry.get("heteronyms", []):
            for defn in het.get("definitions", []):
                for ex in defn.get("example", []):
                    if _FFF9 not in ex or _FFFB not in ex:
                        continue
                    s = ex.index(_FFF9) + 1
                    e = ex.index(_FFFB)
                    hakka = ex[s:e].strip()
                    mand  = ex[e+1:].strip()
                    if "□" in hakka and title and "□" not in title:
                        hakka = hakka.replace("□", title)
                    if "□" in hakka:       skipped["square"] += 1; continue
                    if not mand:           skipped["no_mand"] += 1; continue
                    if not _clean(hakka) or not _clean(mand):
                        skipped["noisy"] += 1; continue
                    key = hakka[:60]
                    if key in seen:        continue
                    seen.add(key)
                    pairs.append(_make("h2m",        hakka, mand,  hakka, mand, "h2m", "dict"))
                    pairs.append(_make("m2h_sixian", mand,  hakka, hakka, mand, "m2h", "dict"))
    print(f"  DICT: {len(pairs)//2} pairs | skipped: {dict(skipped)}")
    return pairs


def extract_lexicon(data_dir):
    pairs = []
    sources = [
        (os.path.join(data_dir, "hat-lexicon-sixian-master", "HAT-Lexicon-Sixian.csv"),
         "m2h_sixian", "hat_sixian"),
        (os.path.join(data_dir, "hat-lexicon-hailu-master",  "HAT-Lexicon-Hailu.csv"),
         "m2h_hailu",  "hat_hailu"),
    ]
    for fp, sys_key, tag in sources:
        if not os.path.exists(fp):
            print(f"  LEX {tag}: not found, skipping"); continue
        with open(fp, "r", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        seen, count = set(), 0
        for row in rows[1:]:
            if len(row) < 3: continue
            mand  = row[0].strip().lstrip("\ufeff")
            hakka = row[2].strip()
            if not mand or not hakka or mand == hakka: continue
            key = f"{tag}:{mand}:{hakka}"
            if key in seen: continue
            seen.add(key)
            pairs.append(_make("h2m",    hakka, mand,  hakka, mand, "h2m", tag))
            pairs.append(_make(sys_key,  mand,  hakka, hakka, mand, "m2h", tag))
            count += 1
        print(f"  LEX {tag}: {count} pairs ({count*2} examples)")
    return pairs


# ── BLEU (pure Python, no deps) ───────────────────────────────────────────────

def _ngrams(tokens, n):
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)) if len(tokens) >= n else Counter()

def sentence_bleu(ref, hyp, max_n=4, smooth=1.0):
    ref, hyp = list(ref.strip()), list(hyp.strip())
    if not hyp: return 0.0
    precs = []
    for n in range(1, max_n+1):
        rc, hc = _ngrams(ref, n), _ngrams(hyp, n)
        clip = total = 0
        for ng, cnt in hc.items():
            clip += min(cnt, rc.get(ng, 0)); total += cnt
        precs.append((clip+smooth)/(total+smooth) if total else 0.0)
    if min(precs) <= 0: return 0.0
    geo = math.exp(sum(math.log(p) for p in precs) / max_n)
    bp  = 1.0 if len(hyp) >= len(ref) else math.exp(1.0 - len(ref)/max(len(hyp), 1))
    return bp * geo

def corpus_bleu(refs, hyps, max_n=4):
    clip_tot, hyp_tot = [0]*max_n, [0]*max_n
    ref_len = hyp_len = 0
    for ref, hyp in zip(refs, hyps):
        rt, ht = list(ref.strip()), list(hyp.strip())
        ref_len += len(rt); hyp_len += len(ht)
        for n in range(1, max_n+1):
            rc, hc = _ngrams(rt, n), _ngrams(ht, n)
            clip = total = 0
            for ng, cnt in hc.items():
                clip += min(cnt, rc.get(ng, 0)); total += cnt
            clip_tot[n-1] += clip; hyp_tot[n-1] += total
    precs = [(c/h if h else 0.0) for c, h in zip(clip_tot, hyp_tot)]
    if min(precs) <= 0 or hyp_len == 0: return 0.0
    geo = math.exp(sum(math.log(p) for p in precs) / max_n)
    bp  = 1.0 if hyp_len >= ref_len else math.exp(1.0 - ref_len/hyp_len)
    return bp * geo


# ── Train / Eval ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── HuggingFace cache ────────────────────────────────────────────────────
    os.makedirs(args.hf_cache, exist_ok=True)
    os.environ["HF_HOME"] = args.hf_cache

    # ── GPU setup ────────────────────────────────────────────────────────────
    # When launched via torchrun, CUDA_VISIBLE_DEVICES must be set externally.
    is_torchrun = "LOCAL_RANK" in os.environ
    if not is_torchrun:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Always derive gpu_ids from the actual env var (set externally or above)
    gpu_ids = [int(x) for x in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")]

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from trl import SFTConfig, SFTTrainer

    n_gpus = torch.cuda.device_count()
    print(f"Visible GPUs: {n_gpus}  (ids: {gpu_ids})")
    if n_gpus > 0:
        for i in range(n_gpus):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  {p.total_memory/1024**3:.1f} GB")

    # ── Data ─────────────────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print("\n=== Extracting data ===")
    sent_pairs, dialogues = extract_tsv(args.dataset_dir)
    sent_pairs += extract_dict(args.dataset_dir)
    lex_pairs = [] if args.no_lexicon else extract_lexicon(args.dataset_dir)

    random.seed(args.seed)
    random.shuffle(sent_pairs)
    split = max(1, int(len(sent_pairs) * args.train_ratio))
    sent_train, sent_val = sent_pairs[:split], sent_pairs[split:]

    all_train = sent_train + dialogues + lex_pairs
    random.shuffle(all_train)
    print(f"\nTrain: {len(all_train):,}  |  Val: {len(sent_val):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n=== Loading model ===")
    adapter = args.adapter_path if args.eval_only else None
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name           = adapter or args.model,
        max_seq_length       = args.max_len,
        dtype                = None,
        load_in_4bit         = True,
    )

    if not args.eval_only:
        model = FastLanguageModel.get_peft_model(
            model,
            r                          = args.lora_r,
            lora_alpha                 = args.lora_alpha,
            target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"],
            lora_dropout               = 0,
            bias                       = "none",
            use_gradient_checkpointing = "unsloth",
            random_state               = args.seed,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable params: {trainable:,}  ({trainable/8072220672*100:.2f}%)")

    tokenizer = get_chat_template(
        tokenizer, chat_template="llama-3",
        mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    )

    def fmt(batch):
        texts = []
        for convs, sys in zip(batch["conversations"], batch["system"]):
            msgs = [{"from": "system", "value": sys}] + convs
            texts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False))
        return {"text": texts}

    train_ds = Dataset.from_list(all_train).map(fmt, batched=True)
    val_ds   = Dataset.from_list(sent_val).map(fmt,  batched=True)

    # ── Train ─────────────────────────────────────────────────────────────────
    if not args.eval_only:
        print("\n=== Training ===")
        eff_accum = max(1, args.grad_accum // n_gpus) if n_gpus > 1 else args.grad_accum

        trainer = SFTTrainer(
            model              = model,
            tokenizer          = tokenizer,
            train_dataset      = train_ds,
            eval_dataset       = val_ds,
            dataset_text_field = "text",
            max_seq_length     = args.max_len,
            packing            = True,   # required when Unsloth enables padding_free (multi-GPU)
            args = SFTConfig(
                per_device_train_batch_size = args.batch,
                gradient_accumulation_steps = eff_accum,
                num_train_epochs            = args.epochs,
                learning_rate               = args.lr,
                fp16                        = not torch.cuda.is_bf16_supported(),
                bf16                        = torch.cuda.is_bf16_supported(),
                logging_steps               = 20,
                eval_strategy               = "epoch",
                save_strategy               = "epoch",
                save_total_limit            = 2,
                load_best_model_at_end      = True,
                optim                       = "adamw_8bit",
                weight_decay                = 0.01,
                lr_scheduler_type           = "cosine",
                warmup_ratio                = 0.05,
                seed                        = args.seed,
                output_dir                  = args.output_dir,
                report_to                   = "none",
                ddp_find_unused_parameters  = False,
            ),
        )
        stats = trainer.train()
        print(f"\nTrain loss : {stats.metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Train time : {stats.metrics['train_runtime']/60:.1f} min")
        print(f"VRAM peak  : {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")

        os.makedirs(args.output_dir, exist_ok=True)
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Adapter saved: {args.output_dir}/")

    # ── Eval ──────────────────────────────────────────────────────────────────
    print("\n=== Evaluation ===")
    FastLanguageModel.for_inference(model)

    def clean(text):
        text = re.split(r"<\|eot_id\|>|<\|im_end\|>", text)[0]
        text = re.sub(r"<\|[^|]+\|>", "", text).replace("\ufffd", "").strip()
        for line in text.splitlines():
            if line.strip(): return line.strip()
        return ""

    def generate_batch(sources, sys_key, batch_size=8):
        results = []
        for i in range(0, len(sources), batch_size):
            batch   = sources[i:i+batch_size]
            prompts = [tokenizer.apply_chat_template(
                [{"from": "system", "value": SYSTEM[sys_key]},
                 {"from": "human",  "value": s}],
                tokenize=False, add_generation_prompt=True) for s in batch]
            enc = tokenizer(prompts, return_tensors="pt",
                            padding=True, truncation=False)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=256,
                                     do_sample=False, repetition_penalty=1.1,
                                     eos_token_id=tokenizer.eos_token_id,
                                     pad_token_id=tokenizer.pad_token_id)
            for j, oid in enumerate(out):
                raw = tokenizer.decode(oid[enc["input_ids"][j].shape[0]:],
                                       skip_special_tokens=False)
                results.append(clean(raw))
        return results

    eval_cfg = [
        ("h2m",        "h2m",        lambda ex: ex["hakka"],    lambda ex: ex["mandarin"]),
        ("m2h_sixian", "m2h",        lambda ex: ex["mandarin"], lambda ex: ex["hakka"]),
    ]

    report = {}
    for sys_key, direction, src_fn, ref_fn in eval_cfg:
        subset = [ex for ex in sent_val if ex.get("direction") == direction]
        if not subset:
            continue
        sources = [src_fn(ex) for ex in subset]
        refs    = [ref_fn(ex) for ex in subset]
        preds   = generate_batch(sources, sys_key)
        valid   = [(r, p) for r, p in zip(refs, preds) if p]
        refs_v, preds_v = zip(*valid) if valid else ([], [])
        avg_bleu  = sum(sentence_bleu(r, p) for r, p in valid) / max(len(valid), 1)
        corp      = corpus_bleu(list(refs_v), list(preds_v))
        report[direction] = {"samples": len(valid),
                             "avg_sentence_bleu": round(avg_bleu, 4),
                             "corpus_bleu":       round(corp, 4)}
        print(f"  [{direction}]  n={len(valid)}  "
              f"avg BLEU={avg_bleu:.4f}  corpus BLEU={corp:.4f}")

    try:
        from bert_score import score as bscore
        print("\nComputing BERTScore...")
        for sys_key, direction, src_fn, ref_fn in eval_cfg:
            subset = [ex for ex in sent_val if ex.get("direction") == direction]
            if not subset or direction not in report:
                continue
            sources = [src_fn(ex) for ex in subset]
            refs    = [ref_fn(ex) for ex in subset]
            preds   = generate_batch(sources, sys_key)
            valid   = [(r, p) for r, p in zip(refs, preds) if p]
            if not valid: continue
            refs_v, preds_v = zip(*valid)
            _, _, F1 = bscore(list(preds_v), list(refs_v),
                              model_type="bert-base-chinese", lang="zh", verbose=False)
            avg_f1 = F1.mean().item()
            report[direction]["bert_score_f1"] = round(avg_f1, 4)
            print(f"  [{direction}]  BERTScore F1 = {avg_f1:.4f}")
    except ImportError:
        print("  bert_score not installed — skipping BERTScore")

    print("\n=== Summary ===")
    print(f"{'Direction':<10} {'N':>6} {'avg BLEU':>10} {'Corpus BLEU':>12} {'BERTScore F1':>13}")
    print("-" * 55)
    for d, r in report.items():
        bert = f"{r.get('bert_score_f1', 0):.4f}"
        print(f"{d:<10} {r['samples']:>6} {r['avg_sentence_bleu']:>10.4f} "
              f"{r['corpus_bleu']:>12.4f} {bert:>13}")

    out_report = os.path.join(args.output_dir, "eval_report.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {out_report}")


if __name__ == "__main__":
    main()


#  CUDA_VISIBLE_DEVICES=0,4,6 torchrun --nproc_per_node=2 train.py --dataset-dir /root/nguyen/research/finetune_llm/dataset --output-dir /root/nguyen/research/finetune_llm/outputs