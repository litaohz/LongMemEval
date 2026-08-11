"""B1 -- replayability check (GO/NO-GO gate for the whole phi pipeline).

WHY THIS EXISTS
---------------
Shapley values are differences between evaluations of overlapping subsets.  If
the same input yields a different score on a re-run, that noise is
indistinguishable from a real marginal contribution, and every phi we compute is
garbage.  So determinism must be measured BEFORE anything else is built.

VERIFIED FACT about the upstream harness (grep -rn "seed" src/ == empty):

    LongMemEval sets `temperature: 0` (run_generation.py:210,366 and
    evaluation/evaluate_qa.py:108) but NEVER sets a seed.

temperature=0 is NOT a determinism guarantee on hosted APIs -- batching,
routing and fp non-associativity all leak nondeterminism.  This script
quantifies the resulting flip rate.

DECISION RULE
-------------
    flip_rate == 0            -> GO, proceed to B2 as-is
    0 < flip_rate <= ~1%      -> GO ONLY IF phi effect sizes are validated to be
                                 much larger than this noise floor; must report
                                 it as a limitation
    flip_rate > ~1%           -> NO-GO on this host until a seed is threaded
                                 through and re-measured

Usage:
    python ssv_scale/replay_check.py --data data/longmemeval_oracle.json \
        --n-questions 20 --n-repeats 5 --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import statistics


def load_questions(path: str, n: int, qtype: str | None) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if qtype:
        data = [d for d in data if d.get("question_type") == qtype]
    return data[:n]


def build_prompt(entry: dict, max_sessions: int) -> str:
    """Render sessions into a single prompt.

    Mirrors the upstream 'orig-session' retriever path: content is fully
    determined by the parallel lists haystack_dates / haystack_session_ids /
    haystack_sessions (run_generation.py:75-77, 92-96).  That is exactly why
    ablation needs ZERO harness changes -- we just drop entries from these
    lists at the data layer.
    """
    parts = []
    dates = entry["haystack_dates"][:max_sessions]
    sessions = entry["haystack_sessions"][:max_sessions]
    for date, session in zip(dates, sessions):
        turns = "\n".join(f"{t['role']}: {t['content']}" for t in session)
        parts.append(f"[{date}]\n{turns}")
    return (
        "I will give you several history chats between you and a user. "
        "Please answer the question based on the relevant chat history.\n\n"
        "History Chats:\n\n" + "\n\n".join(parts) +
        f"\n\nCurrent Date: {entry['question_date']}\n"
        f"Question: {entry['question']}\nAnswer:"
    )


def call_model(prompt: str, model: str, seed: int | None) -> str:
    from openai import OpenAI

    client = OpenAI()
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 128,
    }
    if seed is not None:
        kwargs["seed"] = seed  # the missing knob upstream never sets
    resp = client.chat.completions.create(**kwargs)
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="longmemeval_oracle.json")
    ap.add_argument("--n-questions", type=int, default=20)
    ap.add_argument("--n-repeats", type=int, default=5)
    ap.add_argument("--max-sessions", type=int, default=8,
                    help="cap sessions per question to keep this gate cheap")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=None,
                    help="pass to compare seeded vs unseeded flip rates")
    ap.add_argument("--question-type", default="multi-session",
                    help="oracle has 500 q; multi-session (133) is our main target")
    ap.add_argument("--out", default="replay_check_result.json")
    args = ap.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set -- refusing to fake results.")

    questions = load_questions(args.data, args.n_questions, args.question_type)
    if not questions:
        raise SystemExit(f"no questions of type {args.question_type!r} in {args.data}")

    records = []
    flipped = 0
    for qi, entry in enumerate(questions):
        prompt = build_prompt(entry, args.max_sessions)
        answers = [call_model(prompt, args.model, args.seed) for _ in range(args.n_repeats)]
        digests = [hashlib.sha256(a.encode()).hexdigest()[:12] for a in answers]
        distinct = len(set(digests))
        if distinct > 1:
            flipped += 1
        records.append({
            "question_id": entry.get("question_id", f"q{qi}"),
            "question_type": entry.get("question_type"),
            "n_distinct": distinct,
            "digests": digests,
            "answers": answers if distinct > 1 else answers[:1],
        })
        print(f"[{qi + 1}/{len(questions)}] distinct={distinct} "
              f"{'FLIP' if distinct > 1 else 'stable'}")

    flip_rate = flipped / len(questions)
    summary = {
        "model": args.model,
        "seed": args.seed,
        "n_questions": len(questions),
        "n_repeats": args.n_repeats,
        "question_type": args.question_type,
        "n_flipped": flipped,
        "flip_rate": flip_rate,
        "mean_distinct": statistics.mean(r["n_distinct"] for r in records),
        "distinct_hist": dict(collections.Counter(r["n_distinct"] for r in records)),
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "records": records}, fh, indent=2, ensure_ascii=False)

    print("\n" + json.dumps(summary, indent=2))
    if flip_rate == 0:
        print("\nGO -- deterministic on this sample. Proceed to B2.")
    elif flip_rate <= 0.01:
        print(f"\nGO WITH CAVEAT -- flip_rate={flip_rate:.2%}. "
              "phi effect sizes MUST be shown to exceed this noise floor, "
              "and it must be reported as a limitation.")
    else:
        print(f"\nNO-GO -- flip_rate={flip_rate:.2%} is too high. "
              "Thread a seed through and re-measure before computing any phi.")


if __name__ == "__main__":
    main()
