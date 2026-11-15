"""Evaluation harness for the RAG Chatbot.

Each entry in eval/eval_questions.json looks like:

    {
        "question": "...",
        "answerable": true,
        "must_contain": ["keyword1", "keyword2"]
    },
    {
        "question": "...",
        "answerable": false
    }

- answerable=true -> PASS if the answer is not "I don't know" and contains every entry of `must_contain`.
    * text keywords ignore case, spaces and commas ("k = 60" matches "k=60");
    * numeric keywords ("5", "2,9", "1.000") must match as a whole number, so "5" does not match "15" or "2,5";
    * an entry can be a list of alternatives, any one of which is enough: ["1.000 ppm", "1000 ppm"].
- answerable=false -> PASS if the chatbot refuses with the "I don't know" message (this measures hallucination resistance).

Entries whose question starts with "TODO" are skipped.

Usage:
    python src/evaluate.py                     # default chunker (recursive)
    python src/evaluate.py --chunker semantic
    python src/evaluate.py --chunker all       # both chunkers
    python src/evaluate.py --limit 10          # first 10 questions (default all)
    python src/evaluate.py --workers 4         # parallel workers (default 8)

A per-question report is written to eval/results/<chunker>.json.
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from pathlib import Path

from chatbot import (
    CHUNKERS,
    DEFAULT_CHUNKER,
    EMBEDDING_MODEL,
    IDK_MESSAGE,
    LLM_MODEL,
    ROOT_DIR,
    TOP_K,
    build_rag_chain,
    check_environment,
)

QUESTIONS_FILE = ROOT_DIR / "eval" / "eval_questions.json"
RESULTS_DIR = ROOT_DIR / "eval" / "results"


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #
def load_questions(path: Path, limit: int | None = None) -> list[dict]:
    try:
        items = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as esc:
        sys.exit(f"Cannot read {path}: {esc}")
    if not isinstance(items, list):
        sys.exit(f"{path} must contain a JSON list of test cases.")

    errors, seen, kept = [], set(), []

    for i, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            errors.append(f"#{i}: not an object")
            continue
        question = item.get("question")
        if not isinstance(question, str) or not question.strip():
            errors.append(f"#{i}: missing or empty 'question'")
            continue
        if question.strip().upper().startswith("TODO"):
            errors.append(f"#{i}: empty question (starts with TODO) ({question[:50]!r})")
            continue
        if not isinstance(item.get("answerable"), bool):
            errors.append(f"#{i}: 'answerable' must be true or false ({question[:50]!r})")
            continue
        keywords = item.get("must_contain", [])
        if not isinstance(keywords, list) or not all(valid_keyword(k) for k in keywords):
            errors.append(f"#{i}: 'must_contain' must be a list of non-empty strings or lists of alternatives ({question[:50]!r})")
            continue
        if item["answerable"] and not keywords:
            errors.append(f"#{i}: answerable question without 'must_contain' ({question[:50]!r})")
            continue
        key = question.strip().lower()
        if key in seen:
            errors.append(f"#{i}: duplicate question ({question[:50]!r})")
            continue
        seen.add(key)
        kept.append(item)

    if errors:
        sys.exit("Invalid eval file:\n    " + "\n    ".join(errors))
    if not kept:
        sys.exit(f"No test cases found in {path}.")

    return kept[:limit] if limit else kept


# --------------------------------------------------------------------------- #
# Judging and statistics
# --------------------------------------------------------------------------- #
def is_refusal(answer: str) -> bool:
    text = answer.lower().replace("’", "'")
    return IDK_MESSAGE.lower().rstrip(".") in text


def valid_keyword(k) -> bool:
    if isinstance(k, str):
        return bool(k.strip())
    return (isinstance(k, list) and bool(k) and all(isinstance(a, str) and a.strip() for a in k))


def _clean(s: str) -> str:
    """Unicode-normalise and lowercase text.

    Vietnamese diacritics can be composed or decomposed.
    """
    return unicodedata.normalize("NFC", s).lower().replace("\u00a0", " ")


NUMBER = re.compile(r"^\d+(?:[.,]\d+)*$")


def normalize(s: str) -> str:
    return re.sub(r"[\s,]", "", _clean(s))


def has_keyword(answer: str, keyword: str | list[str]) -> bool:
    if isinstance(keyword, list):
        return any(has_keyword(answer, alt) for alt in keyword)
    kw = _clean(keyword).strip()
    if NUMBER.match(kw):
        # whole-number match: not glued to other digits/letters or to a decimal part
        pattern = rf"(?<!\w)(?<!\d[.,]){re.escape(kw)}(?!\d)(?![.,]\d)"
        return re.search(pattern, _clean(answer)) is not None
    return normalize(keyword) in normalize(answer)


def judge(item: dict, answer: str) -> tuple[bool, str]:
    """Return (passed, reason_if_failed)."""
    refused = is_refusal(answer)
    if item["answerable"]:
        missing = [
            k
            for k in item.get("must_contain", [])
            if not has_keyword(answer, k)
        ]
        if refused:
            ok, reason = False, "refused although the answer is in the papers"
        elif missing:
            ok, reason = False, f"missing keywords: {missing}"
        else:
            ok, reason = True, ""
    else:
        ok = refused
        reason = "" if ok else "answered instead of refusing"

    return ok, reason


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def fmt(k: int, n: int, ci: bool = False) -> str:
    if n == 0:
        return "n/a"
    text = f"{k}/{n} ({k / n:.0%})"
    if ci:
        low, high = wilson(k, n)
        text += f", 95% CI {low:.0%}-{high:.0%}"
    return text


def summarize(results: list[dict]) -> dict:
    def count(cases):
        return sum(case["pass"] for case in cases), len(cases)

    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    return {
        "answerable": count(answerable),
        "unanswerable": count(unanswerable),
        "overall": count(results),
    }


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #
def run_chunker(chunker: str, items: list[dict], rebuild: bool, workers: int) -> list[dict]:
    chain, _ = build_rag_chain(chunker, rebuild)
    questions = [i["question"] for i in items]
    outputs: list = [None] * len(items)

    print(f"\n=== chunker='{chunker}' | {len(items)} questions | {workers} parallel ===")

    for done, (idx, output) in enumerate(
        chain.batch_as_completed(
            questions,
            config={"max_concurrency": workers},
            return_exceptions=True,
        ),
        start=1,
    ):
        outputs[idx] = output
        print(f"\r    answered {done}/{len(items)}", end="", flush=True)
    print()

    results = []

    for item, output in zip(items, outputs):
        if isinstance(output, Exception):
            results.append(
                {
                    "question": item["question"],
                    "answerable": item["answerable"],
                    "must_contain": item.get("must_contain", []),
                    "pass": False,
                    "answer": "",
                    "reason": f"error: {type(output).__name__}: {output}",
                }
            )
            continue
        ok, reason = judge(item, output["answer"])
        results.append(
            {
                "question": item["question"],
                "answerable": item["answerable"],
                "must_contain": item.get("must_contain", []),
                "pass": ok,
                "answer": output["answer"],
                "reason": reason,
            }
        )

    return results


def report(chunker: str, results: list[dict]) -> dict:
    summary = summarize(results)

    failures = [(i, f) for i, f in enumerate(results, start=1) if not f["pass"]]
    if failures:
        print(f"\nFailed cases ({len(failures)}):")
        for i, f in failures:
            kind = "answerable" if f["answerable"] else "unanswerable"
            print(f"[{i:03d}] ({kind}) {f['question']}")
            print(f"        answer : {f['answer'][:200]!r}")
            print(f"        reason : {f['reason']}")

    print(f"\n--- summary: {chunker} ---")
    print(f"      answerable   : {fmt(*summary['answerable'], ci=True)}")
    print(f"      unanswerable : {fmt(*summary['unanswerable'], ci=True)}")
    print(f"      overall      : {fmt(*summary['overall'], ci=True)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = RESULTS_DIR / f"{chunker}.json"
    output_file.write_text(
        json.dumps(
            {
                "meta": {
                    "chunker": chunker,
                    "embedding_model": EMBEDDING_MODEL,
                    "llm_model": LLM_MODEL,
                    "top_k": TOP_K,
                    "num_cases": len(results),
                },
                "summary": {
                    k: {"passed": v[0], "total": v[1]} for k, v in summary.items()
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    
    return summary


def print_comparison(all_summaries: dict[str, dict]) -> None:
    print("\n" + "=" * 50)
    print()

    headers = ("Chunker", "Answerable", "Unanswerable", "Overall")

    rows = [
        (
            chunker.capitalize(),
            fmt(*summary["answerable"]),
            fmt(*summary["unanswerable"]),
            fmt(*summary["overall"]),
        )
        for chunker, summary in all_summaries.items()
    ]
    widths = [
        max(len(str(value))
        for value in column)
        for column in zip(headers, *rows)
    ]

    def print_row(row):
        print(" | ".join(str(value).ljust(width) for value, width in zip(row, widths)))
    print_row(headers)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print_row(row)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG Chatbot")
    parser.add_argument(
        "--chunker",
        choices=(*CHUNKERS, "all"),
        default=DEFAULT_CHUNKER,
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force re-building the index",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="only run the first N test cases",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="questions answered in parallel",
    )
    args = parser.parse_args()

    check_environment()

    items = load_questions(QUESTIONS_FILE, args.limit)
    n_ans = sum(i["answerable"] for i in items)
    nn_ans = len(items) - n_ans
    print(f"\nLoaded {len(items)} test cases ({n_ans} answerable, {nn_ans} unanswerable).")
    print()

    chunkers = CHUNKERS if args.chunker == "all" else (args.chunker,)
    all_results, all_summaries = {}, {}
    for chunker in chunkers:
        all_results[chunker] = run_chunker(chunker, items, args.rebuild, max(1, args.workers))
        all_summaries[chunker] = report(chunker, all_results[chunker])

    if len(chunkers) > 1:
        print_comparison(all_summaries)


if __name__ == "__main__":
    main()
