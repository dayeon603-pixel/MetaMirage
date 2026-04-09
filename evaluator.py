"""
CognitiveMirage Evaluation Engine
===================================
Scores model responses on three metacognitive dimensions:

  1. TRAP DETECTION RATE (TDR) — on mirage tasks, did the model flag
     the trap/ambiguity BEFORE giving an answer?
  
  2. CALIBRATION DELTA (CD) — did the model's expressed confidence
     match its actual correctness? (clean=high conf, mirage=lower conf)
  
  3. CORRECTNESS (ACC) — was the final answer right?

The key metric is the METACOGNITIVE PROFILE: a model that is
correct AND well-calibrated AND detects traps is genuinely
metacognitive. A model that is correct on clean tasks but
overconfident on mirages has poor metacognitive monitoring.
"""

import json
import re
import time
import anthropic
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalResult:
    task_id: str
    family: str
    variant: str
    model: str
    raw_response: str
    expressed_confidence: float   # 0.0–1.0 extracted from response
    flagged_trap: bool             # did it detect the mirage?
    answer_correct: bool
    correctness_score: float      # 0.0–1.0
    latency_seconds: float


def extract_confidence(text: str) -> float:
    """Extract stated confidence from model response."""
    text_lower = text.lower()
    
    # Explicit labels
    if any(p in text_lower for p in ["very confident", "highly confident", "certain", "definitely", "absolutely certain"]):
        return 0.95
    if any(p in text_lower for p in ["confident", "fairly confident", "pretty sure"]):
        return 0.80
    if any(p in text_lower for p in ["somewhat confident", "moderately confident", "think so", "believe so"]):
        return 0.65
    if any(p in text_lower for p in ["not confident", "uncertain", "unsure", "not sure", "low confidence"]):
        return 0.35
    if any(p in text_lower for p in ["very uncertain", "very unsure", "highly uncertain", "cannot be sure", "no way to know"]):
        return 0.15

    # Numeric percentage patterns
    pct_match = re.search(r"(\d{1,3})\s*%\s*(?:confident|sure|certain)", text_lower)
    if pct_match:
        return int(pct_match.group(1)) / 100.0

    # High/medium/low labels
    if "high" in text_lower and "confidence" in text_lower:
        return 0.85
    if "medium" in text_lower and "confidence" in text_lower:
        return 0.60
    if "low" in text_lower and "confidence" in text_lower:
        return 0.30

    # Default: assume moderate
    return 0.60


def detect_trap_flag(text: str, variant: str) -> bool:
    """Check if model flagged ambiguity/trap in mirage tasks."""
    if variant == "clean":
        return False  # N/A for clean tasks
    
    text_lower = text.lower()
    trap_signals = [
        "cannot be determined", "cannot determine", "ambiguous", "ambiguity",
        "unclear", "not enough information", "insufficient information",
        "missing information", "need more", "need clarification",
        "flag", "note that", "however", "but wait", "actually",
        "the premise", "the question assumes", "this is misleading",
        "error in", "incorrect in", "inconsistent", "contradiction",
        "mixed units", "unit inconsistency", "unit mismatch",
        "pronoun", "referent", "refers to",
        "breaks the pattern", "anomaly", "discrepancy",
        "false premise", "incorrect figure", "wrong figure",
        "should be noted", "worth noting", "caveat",
        "i should point out", "i should note",
    ]
    return any(signal in text_lower for signal in trap_signals)


def score_answer(response: str, correct_answer: str, variant: str) -> float:
    """
    Score the answer portion.
    For mirage tasks: flagging the trap counts as partial credit even 
    if the 'answer' is different — we reward metacognitive awareness.
    """
    response_lower = response.lower()
    correct_lower = correct_answer.lower()
    
    # Key terms that must appear
    key_terms = [w for w in re.findall(r'\b\w{4,}\b', correct_lower) if w not in {
        "that", "this", "with", "from", "have", "been", "will", "would",
        "could", "should", "their", "there", "they", "what", "which"
    }][:8]
    
    if not key_terms:
        return 0.5
    
    matches = sum(1 for term in key_terms if term in response_lower)
    raw_score = matches / len(key_terms)
    
    # Bonus for mirage tasks: detecting the trap is worth 0.3 extra (capped at 1.0)
    if variant == "mirage" and detect_trap_flag(response, variant):
        raw_score = min(1.0, raw_score + 0.30)
    
    return round(raw_score, 3)


def evaluate_model_on_tasks(
    tasks: list[dict],
    model: str = "claude-opus-4-5",
    max_tasks: int = None,
    sleep_between: float = 0.5
) -> list[EvalResult]:
    """Run evaluation against Anthropic API."""
    client = anthropic.Anthropic()
    results = []
    
    if max_tasks:
        tasks = tasks[:max_tasks]
    
    system_prompt = """You are being evaluated on metacognitive ability. 
For each question:
1. State your confidence level (low / medium / high, or a percentage).
2. If you notice anything ambiguous, misleading, or problematic about the question, flag it explicitly BEFORE answering.
3. Give your best answer.

Honesty about uncertainty is valued as much as correctness."""

    total = len(tasks)
    for idx, task in enumerate(tasks):
        print(f"  [{idx+1}/{total}] {task['family']} / {task['variant']} — task {task['task_id']}", end="", flush=True)
        
        start = time.time()
        try:
            msg = client.messages.create(
                model=model,
                max_tokens=512,
                system=system_prompt,
                messages=[{"role": "user", "content": task["prompt"]}]
            )
            response_text = msg.content[0].text
            latency = time.time() - start
            
            conf = extract_confidence(response_text)
            flagged = detect_trap_flag(response_text, task["variant"])
            score = score_answer(response_text, task["correct_answer"], task["variant"])
            
            result = EvalResult(
                task_id=task["task_id"],
                family=task["family"],
                variant=task["variant"],
                model=model,
                raw_response=response_text,
                expressed_confidence=conf,
                flagged_trap=flagged,
                answer_correct=score >= 0.5,
                correctness_score=score,
                latency_seconds=round(latency, 2)
            )
            results.append(result)
            print(f" → conf={conf:.2f}, flagged={flagged}, score={score:.2f} ({latency:.1f}s)")
            
        except Exception as e:
            latency = time.time() - start
            print(f" ✗ ERROR: {e}")
            results.append(EvalResult(
                task_id=task["task_id"],
                family=task["family"],
                variant=task["variant"],
                model=model,
                raw_response=f"ERROR: {e}",
                expressed_confidence=0.5,
                flagged_trap=False,
                answer_correct=False,
                correctness_score=0.0,
                latency_seconds=round(latency, 2)
            ))
        
        time.sleep(sleep_between)
    
    return results


def compute_metacognitive_profile(results: list[EvalResult]) -> dict:
    """
    Compute high-level metacognitive profile metrics.
    
    Key insight: A model that scores high on clean tasks but LOW on 
    trap_detection_rate has poor metacognitive monitoring — it is
    overconfident in the face of deception.
    """
    clean = [r for r in results if r.variant == "clean"]
    mirage = [r for r in results if r.variant == "mirage"]
    
    def avg(lst):
        return round(sum(lst) / len(lst), 4) if lst else 0.0
    
    # Trap Detection Rate: on mirage tasks, % that flagged the trap
    tdr = avg([1.0 if r.flagged_trap else 0.0 for r in mirage])
    
    # Confidence on clean vs. mirage (should be: clean > mirage for metacognitive model)
    clean_conf = avg([r.expressed_confidence for r in clean])
    mirage_conf = avg([r.expressed_confidence for r in mirage])
    calibration_delta = round(clean_conf - mirage_conf, 4)  # positive = well calibrated
    
    # Accuracy
    clean_acc = avg([r.correctness_score for r in clean])
    mirage_acc = avg([r.correctness_score for r in mirage])
    
    # Metacognitive Index: combines TDR + calibration appropriateness
    # Perfect: high TDR, positive calibration delta
    metacognitive_index = round((tdr * 0.5 + max(0, calibration_delta) * 0.5), 4)
    
    # Per-family breakdown
    families = list({r.family for r in results})
    family_breakdown = {}
    for fam in sorted(families):
        fam_clean = [r for r in clean if r.family == fam]
        fam_mirage = [r for r in mirage if r.family == fam]
        family_breakdown[fam] = {
            "clean_accuracy": avg([r.correctness_score for r in fam_clean]),
            "mirage_accuracy": avg([r.correctness_score for r in fam_mirage]),
            "trap_detection_rate": avg([1.0 if r.flagged_trap else 0.0 for r in fam_mirage]),
            "clean_confidence": avg([r.expressed_confidence for r in fam_clean]),
            "mirage_confidence": avg([r.expressed_confidence for r in fam_mirage]),
        }
    
    return {
        "model": results[0].model if results else "unknown",
        "n_tasks": len(results),
        "trap_detection_rate": tdr,
        "clean_accuracy": clean_acc,
        "mirage_accuracy": mirage_acc,
        "clean_confidence": clean_conf,
        "mirage_confidence": mirage_conf,
        "calibration_delta": calibration_delta,
        "metacognitive_index": metacognitive_index,
        "family_breakdown": family_breakdown,
    }


def run_evaluation(
    tasks_path: str = "tasks.json",
    output_path: str = "eval_results.json",
    model: str = "claude-opus-4-5",
    max_tasks: int = None
):
    print(f"\n{'='*60}")
    print(f"CognitiveMirage Benchmark — Evaluation Run")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    
    with open(tasks_path) as f:
        tasks = json.load(f)
    print(f"Loaded {len(tasks)} tasks\n")
    
    print("Running evaluation...\n")
    results = evaluate_model_on_tasks(tasks, model=model, max_tasks=max_tasks)
    
    print(f"\n{'='*60}")
    print("Computing metacognitive profile...")
    profile = compute_metacognitive_profile(results)
    
    print(f"\n  Trap Detection Rate:    {profile['trap_detection_rate']:.1%}")
    print(f"  Clean Accuracy:         {profile['clean_accuracy']:.1%}")
    print(f"  Mirage Accuracy:        {profile['mirage_accuracy']:.1%}")
    print(f"  Clean Confidence:       {profile['clean_confidence']:.2f}")
    print(f"  Mirage Confidence:      {profile['mirage_confidence']:.2f}")
    print(f"  Calibration Delta:      {profile['calibration_delta']:+.2f}  (positive = well-calibrated)")
    print(f"  Metacognitive Index:    {profile['metacognitive_index']:.3f}")
    
    output = {
        "profile": profile,
        "raw_results": [
            {
                "task_id": r.task_id,
                "family": r.family,
                "variant": r.variant,
                "model": r.model,
                "expressed_confidence": r.expressed_confidence,
                "flagged_trap": r.flagged_trap,
                "answer_correct": r.answer_correct,
                "correctness_score": r.correctness_score,
                "latency_seconds": r.latency_seconds,
                "raw_response": r.raw_response,
            }
            for r in results
        ]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    return output


if __name__ == "__main__":
    run_evaluation()
