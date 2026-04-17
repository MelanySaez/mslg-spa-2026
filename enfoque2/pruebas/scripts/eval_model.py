"""Evalúa modelo fine-tuneado sobre test set. Reutilizable en Ronda 1 y Ronda 2.

Acepta CSV con columnas:
  - source + target  → genera predicciones + métricas (BLEU, ROUGE)
  - source only      → genera predicciones sin métricas (test ciego)

También acepta TSV de competencia (--src-col, --tgt-col para mapear columnas).
"""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate as hf_evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_beams", type=int, default=4)
    p.add_argument("--max_src_len", type=int, default=128)
    p.add_argument("--max_tgt_len", type=int, default=128)
    p.add_argument("--num_examples", type=int, default=20)
    p.add_argument("--src-col", default="source",
                   help="Nombre columna fuente (default: source)")
    p.add_argument("--tgt-col", default="target",
                   help="Nombre columna referencia (default: target; omitir si test ciego)")
    p.add_argument("--sep", default=",",
                   help="Separador del archivo (default: ',' ; usar '\\t' para TSV)")
    return p.parse_args()


def main():
    args = parse_args()
    sep = "\t" if args.sep == "\\t" else args.sep

    print(f"[Eval] Cargando modelo: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"[Eval] Dispositivo: {device}")

    print(f"[Eval] Cargando test: {args.test_csv}")
    df = pd.read_csv(args.test_csv, sep=sep)
    df = df.dropna(subset=[args.src_col])
    print(f"[Eval] Registros: {len(df)}")

    sources = df[args.src_col].astype(str).tolist()

    has_targets = args.tgt_col in df.columns
    targets = df[args.tgt_col].astype(str).tolist() if has_targets else []

    if not has_targets:
        print("[Eval] Sin columna de referencia → modo predicción-only (sin métricas)")

    all_preds = []
    print("[Eval] Generando predicciones...")
    for i in range(0, len(sources), args.batch_size):
        batch_src = sources[i:i + args.batch_size]
        inputs = tokenizer(
            batch_src,
            max_length=args.max_src_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                num_beams=args.num_beams,
                max_length=args.max_tgt_len,
                early_stopping=True,
            )
        preds = tokenizer.batch_decode(generated, skip_special_tokens=True)
        all_preds.extend([p.strip() for p in preds])
        if (i // args.batch_size) % 5 == 0:
            print(f"   {min(i + args.batch_size, len(sources))}/{len(sources)}")

    metrics = {"total_samples": len(all_preds)}

    if has_targets:
        print("[Eval] Calculando métricas...")
        bleu = hf_evaluate.load("sacrebleu")
        rouge = hf_evaluate.load("rouge")
        bleu_r = bleu.compute(predictions=all_preds, references=[[r] for r in targets])
        rouge_r = rouge.compute(predictions=all_preds, references=targets)
        metrics.update({
            "bleu": round(bleu_r["score"], 4),
            "bleu_bp": round(bleu_r["bp"], 4),
            "rouge1": round(rouge_r["rouge1"], 4),
            "rouge2": round(rouge_r["rouge2"], 4),
            "rougeL": round(rouge_r["rougeL"], 4),
        })
        print("\n── Resultados ──")
        for k, v in metrics.items():
            print(f"   {k:16s}: {v}")

    examples = []
    for i in range(min(args.num_examples, len(all_preds))):
        entry = {"source": sources[i], "prediction": all_preds[i]}
        if has_targets:
            entry["reference"] = targets[i]
            entry["exact_match"] = all_preds[i].strip() == targets[i].strip()
        examples.append(entry)

    ids = df["ID"].astype(str).tolist() if "ID" in df.columns else [str(i) for i in range(len(all_preds))]

    output = {
        "model": args.model_dir,
        "test_csv": args.test_csv,
        "metrics": metrics,
        "predictions": [{"id": ids[i], "source": sources[i], "prediction": all_preds[i]}
                        for i in range(len(all_preds))],
        "examples": examples,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n[Eval] Guardado: {args.output_file}")


if __name__ == "__main__":
    main()
