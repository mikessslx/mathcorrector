import argparse
import json
from collections import Counter
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    return text.strip().split()


def exact_match(predictions: list[str], targets: list[str]) -> float:
    if not predictions:
        return 0.0
    matches = sum(1 for p, t in zip(predictions, targets) if p.strip() == t.strip())
    return matches / len(predictions)


def token_level_prf(
    sources: list[str], predictions: list[str], targets: list[str], beta: float = 0.5
) -> dict:
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for src, pred, tgt in zip(sources, predictions, targets):
        src_tokens = _tokenize(src)
        pred_tokens = _tokenize(pred)
        tgt_tokens = _tokenize(tgt)

        gold_edits = _get_edits(src_tokens, tgt_tokens)
        pred_edits = _get_edits(src_tokens, pred_tokens)

        tp = len(gold_edits & pred_edits)
        fp = len(pred_edits - gold_edits)
        fn = len(gold_edits - pred_edits)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)

    return {"precision": precision, "recall": recall, f"f_{beta}": f_beta}


def _get_edits(src_tokens: list[str], tgt_tokens: list[str]) -> set[tuple]:
    edits = set()
    sm = _simple_align(src_tokens, tgt_tokens)
    for op, i1, i2, j1, j2 in sm:
        if op != "equal":
            src_span = " ".join(src_tokens[i1:i2])
            tgt_span = " ".join(tgt_tokens[j1:j2])
            edits.add((i1, i2, src_span, tgt_span))
    return edits


def _simple_align(src: list[str], tgt: list[str]) -> list[tuple]:
    import difflib

    matcher = difflib.SequenceMatcher(None, src, tgt)
    return matcher.get_opcodes()


def gleu_score(sources: list[str], predictions: list[str], targets: list[str]) -> float:
    """GLEU: reward correct edits, penalize bad edits and missed edits."""
    total_gleu = 0.0

    for src, pred, tgt in zip(sources, predictions, targets):
        src_tokens = _tokenize(src)
        pred_tokens = _tokenize(pred)
        tgt_tokens = _tokenize(tgt)

        def ngrams(tokens, n):
            return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))

        max_n = min(4, min(len(pred_tokens), len(tgt_tokens)))
        if max_n == 0:
            continue

        log_gleu = 0.0
        for n in range(1, max_n + 1):
            pred_ng = ngrams(pred_tokens, n)
            tgt_ng = ngrams(tgt_tokens, n)
            src_ng = ngrams(src_tokens, n)

            # GLEU: correct n-grams = intersection with target but not just copied from source
            correct = sum((pred_ng & tgt_ng).values())
            total = sum(pred_ng.values())

            if total == 0:
                continue
            log_gleu += max(0, _safe_log(correct / total)) / max_n

        import math
        bp = min(1.0, math.exp(1 - len(tgt_tokens) / max(len(pred_tokens), 1)))
        total_gleu += bp * math.exp(log_gleu)

    return total_gleu / max(len(sources), 1)


def _safe_log(x: float) -> float:
    import math

    return math.log(max(x, 1e-10))


def character_error_rate(predictions: list[str], targets: list[str]) -> float:
    total_cer = 0.0
    for pred, tgt in zip(predictions, targets):
        distance = _levenshtein(pred, tgt)
        total_cer += distance / max(len(tgt), 1)
    return total_cer / max(len(predictions), 1)


def _levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


def detection_accuracy(
    sources: list[str], predictions: list[str], targets: list[str]
) -> dict:
    """Measures whether the model correctly identifies WHICH sentences need correction."""
    tp = fp = tn = fn = 0

    for src, pred, tgt in zip(sources, predictions, targets):
        has_error = (src.strip() != tgt.strip())
        model_changed = (src.strip() != pred.strip())

        if has_error and model_changed:
            tp += 1
        elif has_error and not model_changed:
            fn += 1
        elif not has_error and model_changed:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    return {
        "detection_accuracy": (tp + tn) / max(total, 1),
        "detection_precision": tp / max(tp + fp, 1),
        "detection_recall": tp / max(tp + fn, 1),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def evaluate_all(
    sources: list[str], predictions: list[str], targets: list[str]
) -> dict:
    results = {}
    results["exact_match"] = exact_match(predictions, targets)
    results["token_prf"] = token_level_prf(sources, predictions, targets, beta=0.5)
    results["gleu"] = gleu_score(sources, predictions, targets)
    results["cer"] = character_error_rate(predictions, targets)
    results["detection"] = detection_accuracy(sources, predictions, targets)
    results["num_samples"] = len(sources)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_file", required=True, help="JSONL with source/target/prediction")
    parser.add_argument("--output_file", default=None, help="Save results as JSON")
    args = parser.parse_args()

    data = []
    with open(args.predictions_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    sources = [d["source"] for d in data]
    targets = [d["target"] for d in data]
    predictions = [d["prediction"] for d in data]

    results = evaluate_all(sources, predictions, targets)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Samples:            {results['num_samples']}")
    print(f"  Exact Match:        {results['exact_match']:.4f}")
    print(f"  Token Precision:    {results['token_prf']['precision']:.4f}")
    print(f"  Token Recall:       {results['token_prf']['recall']:.4f}")
    print(f"  Token F_0.5:        {results['token_prf']['f_0.5']:.4f}")
    print(f"  GLEU:               {results['gleu']:.4f}")
    print(f"  CER:                {results['cer']:.4f}")
    print(f"  Detection Acc:      {results['detection']['detection_accuracy']:.4f}")
    print(f"  Detection Prec:     {results['detection']['detection_precision']:.4f}")
    print(f"  Detection Recall:   {results['detection']['detection_recall']:.4f}")
    print("=" * 60)

    if args.output_file:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")


if __name__ == "__main__":
    main()
