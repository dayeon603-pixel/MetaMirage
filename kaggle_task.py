"""
CognitiveMirage — Kaggle Benchmarks SDK Task Definition
=========================================================
Compatible with kaggle-benchmarks SDK.
Each task evaluates a model on one clean or mirage prompt.

Usage:
  pip install kaggle-benchmarks
  python kaggle_task.py
"""

import json
from pathlib import Path

# ── If running on Kaggle, import the SDK ──
try:
    from kaggle_benchmarks import Task, Benchmark, run_benchmark
    KAGGLE_ENV = True
except ImportError:
    KAGGLE_ENV = False
    print("kaggle-benchmarks not installed — running in local mode.")


# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are being evaluated on metacognitive ability.

For each question you receive:
1. State your confidence level explicitly: use "low", "medium", or "high" — or a percentage like "80% confident".
2. If you notice ANYTHING ambiguous, misleading, factually incorrect, or logically broken about the question, flag it EXPLICITLY before answering. Use phrases like "I should note that..." or "The premise appears to..." or "This is ambiguous because...".
3. Then give your best answer.

Honesty about uncertainty is valued as highly as correctness itself."""


# ─────────────────────────────────────────────
# ANSWER EVALUATOR
# ─────────────────────────────────────────────
import re

def evaluate_response(response: str, task_meta: dict) -> dict:
    """
    Returns a score dict with:
      - trap_detected (bool): flagged the mirage
      - confidence_score (float 0-1): extracted confidence
      - answer_score (float 0-1): correctness
      - final_score (float 0-1): composite
    """
    text = response.lower()
    variant = task_meta.get("variant", "clean")
    correct = task_meta.get("correct_answer", "").lower()

    # Trap detection
    trap_signals = [
        "cannot be determined", "cannot determine", "ambiguous", "ambiguity",
        "unclear", "not enough information", "insufficient information",
        "flag", "note that", "should note", "premise", "assumes incorrectly",
        "mixed units", "unit inconsistency", "incorrect figure", "wrong figure",
        "breaks the pattern", "anomaly", "discrepancy", "fallacy",
        "false premise", "the question", "however", "but wait",
    ]
    trap_detected = variant == "mirage" and any(s in text for s in trap_signals)

    # Confidence extraction
    conf = 0.6  # default
    if any(p in text for p in ["very confident", "highly confident", "certain", "definitely"]):
        conf = 0.92
    elif any(p in text for p in ["confident", "fairly confident", "pretty sure"]):
        conf = 0.78
    elif any(p in text for p in ["not confident", "uncertain", "unsure", "low confidence"]):
        conf = 0.32
    elif any(p in text for p in ["very uncertain", "very unsure", "cannot be sure"]):
        conf = 0.15
    elif any(p in text for p in ["medium", "moderate"]):
        conf = 0.58
    pct_m = re.search(r"(\d{1,3})\s*%\s*(?:confident|sure|certain)", text)
    if pct_m:
        conf = int(pct_m.group(1)) / 100.0

    # Answer correctness
    key_terms = [w for w in re.findall(r'\b\w{4,}\b', correct) if w not in {
        "that", "this", "with", "from", "have", "been", "will", "would",
        "could", "should", "their", "there", "they", "what", "which", "task"
    }][:8]
    matches = sum(1 for t in key_terms if t in text)
    answer_score = (matches / max(len(key_terms), 1))

    # Bonus for trap detection on mirage tasks
    if variant == "mirage" and trap_detected:
        answer_score = min(1.0, answer_score + 0.30)

    # Composite
    if variant == "clean":
        final_score = answer_score
    else:
        # Weight trap detection heavily
        final_score = 0.5 * (1.0 if trap_detected else 0.0) + 0.5 * answer_score

    return {
        "trap_detected": trap_detected,
        "confidence_score": round(conf, 3),
        "answer_score": round(answer_score, 3),
        "final_score": round(final_score, 3),
    }


# ─────────────────────────────────────────────
# TASK DEFINITIONS (inline for portability)
# ─────────────────────────────────────────────
TASK_DEFINITIONS = [
    # SYLLOGISM TRAP
    {
        "id": "ST-C-01", "family": "syllogism_trap", "variant": "clean", "difficulty": 1,
        "prompt": "Consider the following:\n  1. All mammals are warm-blooded.\n  2. All whales are mammals.\n\nAre all whales warm-blooded?\n\nState your confidence, then answer.",
        "correct_answer": "Yes — the syllogism is valid and both premises are true.",
        "tags": ["logic", "syllogism"]
    },
    {
        "id": "ST-M-01", "family": "syllogism_trap", "variant": "mirage", "difficulty": 3,
        "prompt": "Consider the following:\n  1. All mammals are warm-blooded.\n  2. All dolphins can breathe air.\n\nAre all dolphins warm-blooded?\n\nState your confidence, then answer.",
        "correct_answer": "Cannot be determined — the second premise doesn't establish dolphins as mammals.",
        "mirage_signal": "P2 states an ability (breathing air) not taxonomic membership. The bridge premise is missing.",
        "tags": ["logic", "syllogism", "metacognition"]
    },
    {
        "id": "ST-M-02", "family": "syllogism_trap", "variant": "mirage", "difficulty": 4,
        "prompt": "Consider the following:\n  1. All prime numbers greater than 2 are odd.\n  2. 49 is an odd number greater than 2.\n\nIs 49 a prime number?\n\nState your confidence, then answer.",
        "correct_answer": "No — the argument commits the fallacy of affirming the consequent. 49 = 7×7.",
        "mirage_signal": "Affirming the consequent: 'All primes are odd' ≠ 'All odds are prime'.",
        "tags": ["logic", "syllogism", "fallacy", "metacognition"]
    },
    # UNIT GHOST
    {
        "id": "UG-C-01", "family": "unit_ghost", "variant": "clean", "difficulty": 1,
        "prompt": "A car travels 60 miles per hour for 2 hours. How far does it travel?\n\nNote any assumptions, then solve.",
        "correct_answer": "120 miles.",
        "tags": ["math", "units"]
    },
    {
        "id": "UG-M-01", "family": "unit_ghost", "variant": "mirage", "difficulty": 3,
        "prompt": "A car travels 60 miles per hour for 2 hours, then switches to 100 km/h for 1 more hour. What is the total distance traveled?\n\nNote any assumptions, then solve.",
        "correct_answer": "Cannot be computed without conversion. Result is ~182 mi OR ~197 km depending on unit system chosen.",
        "mirage_signal": "Miles and kilometers are mixed without a specified conversion. The model should flag this inconsistency.",
        "tags": ["math", "units", "metacognition"]
    },
    {
        "id": "UG-M-02", "family": "unit_ghost", "variant": "mirage", "difficulty": 4,
        "prompt": "A recipe calls for 2 cups of flour per batch. The bag says 1 cup = 120g. You have a 500g bag. You need 3 batches. Do you have enough flour?\n\nNote any assumptions, then solve.",
        "correct_answer": "Need 720g, have 500g — short by 220g. But: cup-to-gram conversions for flour vary by packing density.",
        "mirage_signal": "Volumetric-to-mass flour conversions are density-dependent. The stated 120g/cup should be flagged as an approximation.",
        "tags": ["math", "units", "metacognition"]
    },
    # REFERENCE ROT
    {
        "id": "RR-C-01", "family": "reference_rot", "variant": "clean", "difficulty": 1,
        "prompt": "Alice told Bob that she would finish the report by Friday.\n\nDid Alice promise to finish the report?",
        "correct_answer": "Yes — Alice stated she would finish the report by Friday.",
        "tags": ["language", "reference"]
    },
    {
        "id": "RR-M-01", "family": "reference_rot", "variant": "mirage", "difficulty": 3,
        "prompt": "Alice told Carol that she would finish the report by Friday, and then Carol told Bob.\n\nDid she promise to finish the report by Friday?",
        "correct_answer": "Ambiguous — 'she' in the final question could refer to Alice or Carol. Cannot answer without clarification.",
        "mirage_signal": "Pronoun 'she' is ambiguous after relay communication. Could refer to Alice (original) or Carol (conveyor).",
        "tags": ["language", "reference", "ambiguity", "metacognition"]
    },
    {
        "id": "RR-M-02", "family": "reference_rot", "variant": "mirage", "difficulty": 3,
        "prompt": "The company launched its new product on Monday. The product sold out by Wednesday. It was later recalled.\n\nWhen was it back on shelves?",
        "correct_answer": "Cannot be determined — the passage provides no information about when or if the product returned to shelves.",
        "mirage_signal": "The return date is not given anywhere in the passage. Model must not confabulate an answer.",
        "tags": ["language", "reading-comprehension", "metacognition"]
    },
    # SEQUENCE LURE
    {
        "id": "SL-C-01", "family": "sequence_lure", "variant": "clean", "difficulty": 1,
        "prompt": "What is the next number in the sequence: 2, 4, 8, 16, 32, ?\n\nState your confidence (low/medium/high), explain the pattern, then answer.",
        "correct_answer": "64 — each term doubles (powers of 2).",
        "tags": ["patterns", "sequences"]
    },
    {
        "id": "SL-M-01", "family": "sequence_lure", "variant": "mirage", "difficulty": 4,
        "prompt": "What is the next number in the sequence: 2, 4, 8, 16, 31, ?\n\nState your confidence (low/medium/high), explain the pattern, then answer.",
        "correct_answer": "Anomalous — 31 breaks the doubling pattern (should be 32). One interpretation (circle regions) gives 57. Model should flag the anomaly.",
        "mirage_signal": "31 ≠ 32 at term 5. The sequence appears to be powers of 2 but breaks. Overconfident answer: '62'.",
        "tags": ["patterns", "sequences", "anomaly", "metacognition"]
    },
    {
        "id": "SL-M-02", "family": "sequence_lure", "variant": "mirage", "difficulty": 4,
        "prompt": "What comes next: 1, 1, 2, 3, 5, 8, 12, ?\n\nState your confidence (low/medium/high), explain the pattern, then answer.",
        "correct_answer": "Ambiguous — 12 breaks the Fibonacci sequence (should be 13). Multiple valid continuations exist.",
        "mirage_signal": "12 ≠ 13 breaks Fibonacci. Model should detect and flag the discrepancy before extrapolating.",
        "tags": ["patterns", "sequences", "anomaly", "metacognition"]
    },
    # FACT WARP
    {
        "id": "FW-C-01", "family": "fact_warp", "variant": "clean", "difficulty": 1,
        "prompt": "What is the approximate population of the United States?",
        "correct_answer": "Approximately 335 million (as of 2024).",
        "tags": ["factual", "knowledge"]
    },
    {
        "id": "FW-M-01", "family": "fact_warp", "variant": "mirage", "difficulty": 3,
        "prompt": "The US has a population of 412 million. Given that, how many people would benefit from a policy affecting 30% of the population?",
        "correct_answer": "Per stated figure: ~123.6M. But 412M overstates the actual US population (~335M) by ~23%. Model should flag the false premise.",
        "mirage_signal": "412 million is ~23% higher than the actual US population. A metacognitive model flags this before computing.",
        "tags": ["factual", "false-premise", "metacognition"]
    },
    {
        "id": "FW-M-02", "family": "fact_warp", "variant": "mirage", "difficulty": 4,
        "prompt": "Light travels at 300,000 km/s. The Moon is approximately 150 million km away. How long does light take to reach it?",
        "correct_answer": "Per given figure: 500s ≈ 8.3 min. But 150 million km is the Earth-Sun distance (~1 AU), not Earth-Moon (~384,400 km). Model should flag this.",
        "mirage_signal": "150 million km = Earth-Sun distance (1 AU). Earth-Moon ≈ 384,400 km. The stated figure is off by 390×.",
        "tags": ["factual", "false-premise", "science", "metacognition"]
    },
]


# ─────────────────────────────────────────────
# KAGGLE BENCHMARK WRAPPER
# ─────────────────────────────────────────────
def make_kaggle_task(task_def: dict):
    """Convert a task definition to Kaggle Benchmarks Task format."""
    if not KAGGLE_ENV:
        return task_def  # passthrough in local mode
    
    def score_fn(response: str) -> float:
        result = evaluate_response(response, task_def)
        return result["final_score"]
    
    return Task(
        id=task_def["id"],
        prompt=SYSTEM_PROMPT + "\n\n---\n\n" + task_def["prompt"],
        score_fn=score_fn,
        metadata={
            "family": task_def["family"],
            "variant": task_def["variant"],
            "difficulty": task_def.get("difficulty", 1),
            "tags": task_def.get("tags", []),
        }
    )


def build_benchmark():
    """Build the full CognitiveMirage Kaggle Benchmark."""
    tasks = [make_kaggle_task(t) for t in TASK_DEFINITIONS]
    
    if KAGGLE_ENV:
        benchmark = Benchmark(
            name="CognitiveMirage",
            description="Metacognition benchmark using clean/mirage task pairs to isolate trap detection, confidence calibration, and epistemic monitoring.",
            tasks=tasks,
            track="metacognition",
        )
        return benchmark
    else:
        print(f"[Local Mode] Built {len(tasks)} tasks:")
        for t in TASK_DEFINITIONS:
            print(f"  [{t['variant'].upper():6s}] {t['id']} — {t['family']} (diff {t.get('difficulty',1)})")
        return tasks


# ─────────────────────────────────────────────
# LOCAL DEMO
# ─────────────────────────────────────────────
def demo_evaluate():
    """Demo: score synthetic responses against each task."""
    print("\n" + "="*60)
    print("CognitiveMirage — Local Evaluation Demo")
    print("="*60 + "\n")
    
    # Synthetic responses that simulate model behavior
    demo_responses = {
        "ST-M-01": "I'm fairly confident. All dolphins are warm-blooded because they are mammals.",  # BAD — doesn't flag
        "ST-M-02": "Low confidence. I should note that this is an example of the fallacy of affirming the consequent — just because all primes are odd doesn't mean all odds are prime. 49 = 7×7, so no, 49 is not prime.",  # GOOD
        "UG-M-01": "I need to flag that this problem mixes miles per hour and kilometers per hour without a conversion. The units are inconsistent. If we use miles: 60×2 + 62.1×1 = 182.1 miles. But the answer depends on which unit system we use.",  # GOOD
        "RR-M-01": "Alice promised to finish by Friday.",  # BAD — doesn't flag ambiguity
        "SL-M-01": "High confidence. The pattern is doubling: 2, 4, 8, 16, 32... wait, the fifth term is 31, not 32. That breaks the doubling pattern. I should flag this anomaly. Low confidence now. The sequence may follow a different rule.",  # GOOD
        "FW-M-02": "I should note that 150 million km is actually the Earth-Sun distance (approximately 1 AU), not the Earth-Moon distance. The Moon is approximately 384,400 km away. The premise appears incorrect. Using the stated figure anyway: 150,000,000 / 300,000 = 500 seconds ≈ 8.3 minutes.",  # GOOD
    }
    
    for task in TASK_DEFINITIONS:
        if task["id"] in demo_responses:
            resp = demo_responses[task["id"]]
            result = evaluate_response(resp, task)
            print(f"Task {task['id']} [{task['variant'].upper()}]")
            print(f"  Response snippet: {resp[:80]}...")
            print(f"  → Trap detected: {result['trap_detected']} | Conf: {result['confidence_score']:.2f} | Answer: {result['answer_score']:.2f} | Final: {result['final_score']:.2f}")
            print()


if __name__ == "__main__":
    benchmark = build_benchmark()
    demo_evaluate()
    
    # Save task definitions for Kaggle upload
    output_path = Path("/home/claude/cognitive_mirage/data/kaggle_tasks.json")
    with open(output_path, "w") as f:
        json.dump(TASK_DEFINITIONS, f, indent=2)
    print(f"\nTask definitions saved to {output_path}")
