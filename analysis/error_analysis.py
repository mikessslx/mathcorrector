import argparse
import difflib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


LATEX_PATTERN = re.compile(r'\\[a-zA-Z]+|[\$\{\}\\^_]|\\\[|\\\]|\\\(|\\\)')
MATH_OPERATORS = {"+", "-", "*", "/", "=", "<", ">", "\\leq", "\\geq", "\\neq", "\\approx", "\\equiv"}
MATH_FUNCTIONS = {"\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\exp", "\\lim", "\\sum", "\\prod", "\\int"}


def classify_edit(src_span: str, tgt_span: str) -> str:
    if not src_span.strip() and tgt_span.strip():
        return "insertion"
    if src_span.strip() and not tgt_span.strip():
        return "deletion"

    if LATEX_PATTERN.search(src_span) or LATEX_PATTERN.search(tgt_span):
        if any(op in src_span or op in tgt_span for op in MATH_OPERATORS):
            return "operator_error"
        if any(fn in src_span or fn in tgt_span for fn in MATH_FUNCTIONS):
            return "function_error"
        return "latex_symbol_error"

    if src_span.replace(" ", "").isdigit() or tgt_span.replace(" ", "").isdigit():
        return "numerical_error"

    return "terminology_error"


def analyze_errors(data: list[dict]) -> dict:
    error_type_counts = Counter()
    error_type_examples = defaultdict(list)
    correction_status = Counter()

    for item in data:
        src = item["source"]
        tgt = item["target"]
        pred = item["prediction"]

        src_tokens = src.strip().split()
        tgt_tokens = tgt.strip().split()
        pred_tokens = pred.strip().split()

        matcher = difflib.SequenceMatcher(None, src_tokens, tgt_tokens)
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == "equal":
                continue

            src_span = " ".join(src_tokens[i1:i2])
            tgt_span = " ".join(tgt_tokens[j1:j2])
            error_type = classify_edit(src_span, tgt_span)
            error_type_counts[error_type] += 1

            pred_matcher = difflib.SequenceMatcher(None, src_tokens, pred_tokens)
            pred_corrected = False
            for pop, pi1, pi2, pj1, pj2 in pred_matcher.get_opcodes():
                if pop != "equal" and pi1 == i1 and pi2 == i2:
                    pred_span = " ".join(pred_tokens[pj1:pj2])
                    if pred_span == tgt_span:
                        pred_corrected = True
                    break

            status = "correct" if pred_corrected else "missed"
            correction_status[(error_type, status)] += 1

            if len(error_type_examples[error_type]) < 5:
                error_type_examples[error_type].append({
                    "source_span": src_span,
                    "target_span": tgt_span,
                    "predicted_span": "N/A",
                    "status": status,
                })

    per_type_accuracy = {}
    for error_type in error_type_counts:
        correct = correction_status.get((error_type, "correct"), 0)
        total = error_type_counts[error_type]
        per_type_accuracy[error_type] = {
            "total": total,
            "correct": correct,
            "accuracy": correct / max(total, 1),
        }

    return {
        "error_type_distribution": dict(error_type_counts),
        "per_type_accuracy": per_type_accuracy,
        "examples": {k: v for k, v in error_type_examples.items()},
    }


def format_report(analysis: dict) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("ERROR ANALYSIS REPORT")
    lines.append("=" * 60)

    lines.append("\n--- Error Type Distribution ---")
    for etype, count in sorted(analysis["error_type_distribution"].items(), key=lambda x: -x[1]):
        lines.append(f"  {etype:25s}: {count:5d}")

    lines.append("\n--- Per-Type Correction Accuracy ---")
    for etype, stats in sorted(analysis["per_type_accuracy"].items(), key=lambda x: -x[1]["total"]):
        lines.append(
            f"  {etype:25s}: {stats['correct']:4d}/{stats['total']:4d} = {stats['accuracy']:.2%}"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", required=True)
    parser.add_argument("--output_file", default=None)
    args = parser.parse_args()

    data = []
    with open(args.predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    analysis = analyze_errors(data)
    print(format_report(analysis))

    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nFull analysis saved to {args.output_file}")


if __name__ == "__main__":
    main()
