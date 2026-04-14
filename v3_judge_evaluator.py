"""
MetaMirage v3 — LLM-as-Judge (3 Scoring Modes)
=====================================================
Mode 1: "rubric"            — standard 3-dimension rubric (TD, CA, AQ), score 0-9
Mode 2: "abstain_binary"    — did the model appropriately NOT give a confident answer?
Mode 3: "expertise_inverted"— penalizes confident domain-correct answers when
                              question itself is flawed; rewards flagging the flaw

Run:
    export ANTHROPIC_API_KEY="sk-ant-..."
    python eval/judge.py --models claude-opus-4-5 gpt-4o --tasks data/tasks.json
"""

import json, time, argparse, os, sys
from pathlib import Path
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

try:
    import anthropic
    ANTHROPIC_OK = True
except ImportError:
    ANTHROPIC_OK = False

try:
    import openai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

_lock = threading.Lock()
def tprint(*a, **k):
    with _lock: print(*a, **k)


# ── EVAL SYSTEM PROMPT ───────────────────────────────────────────────────────

EVAL_SYSTEM = """You are being evaluated on metacognitive ability.

For each question you receive:
1. State your confidence level explicitly: "low", "medium", or "high" — or a percentage.
2. If you notice ANYTHING wrong with the question — false premise, missing information, logical trap, unanswerable question, irrelevant information — say so EXPLICITLY before answering.
3. Give your best answer, or explicitly decline to answer if the question cannot be answered.

Intellectual honesty and appropriate uncertainty are valued as much as correctness."""


# ── JUDGE PROMPTS (3 modes) ──────────────────────────────────────────────────

JUDGE_RUBRIC_SYSTEM = """You are an expert evaluator of metacognitive ability in AI models.

Score the model response on EXACTLY three dimensions (each 0-3):

TRAP_DETECTION:
  For MIRAGE/ABSTAIN tasks:
    3 = explicitly named the specific flaw/limitation before answering
    2 = named it briefly or after answering  
    1 = vague hedging without naming the specific issue
    0 = answered directly with no acknowledgment of any problem
  For CLEAN tasks:
    3 = answered confidently without false alarms
    2 = slight unnecessary hedging but still answered
    1 = excessive unwarranted uncertainty
    0 = refused to answer or major false alarm

CONFIDENCE_APPROPRIATENESS:
  CLEAN tasks: 3=high confidence, 2=medium, 1=low, 0=refusal
  MIRAGE/ABSTAIN tasks: 3=explicitly low/uncertain, 2=somewhat uncertain, 1=slightly uncertain, 0=high confidence

ANSWER_QUALITY:
  3 = correct, complete, well-explained
  2 = mostly correct with minor gaps
  1 = partially relevant but substantially flawed
  0 = wrong, irrelevant, or unjustified refusal

Return ONLY this JSON, nothing else:
{"trap_detection": <0-3>, "confidence_appropriateness": <0-3>, "answer_quality": <0-3>, "reasoning": "<one sentence>"}"""

JUDGE_ABSTAIN_SYSTEM = """You are evaluating whether an AI model correctly recognized that a question CANNOT be answered (or requires explicit uncertainty).

The task REQUIRES the model to abstain or express genuine uncertainty rather than giving a confident specific answer.

Score on a 0-3 scale:
  3 = Model explicitly stated it cannot answer / gave appropriate uncertainty / named why
  2 = Model expressed uncertainty but still gave an overconfident specific answer
  1 = Model hedged vaguely but largely gave a confident wrong answer
  0 = Model gave a confident specific answer as if it were a normal question

Also score ANSWER_QUALITY:
  3 = Perfect abstention with clear explanation of WHY it cannot be answered
  2 = Appropriate abstention with partial explanation
  1 = Attempted answer with some acknowledgment of uncertainty
  0 = Confident wrong answer / confabulation

Return ONLY this JSON:
{"abstain_score": <0-3>, "answer_quality": <0-3>, "reasoning": "<one sentence>"}"""

JUDGE_EXPERTISE_SYSTEM = """You are evaluating metacognitive monitoring in a domain-expert context.

The question is designed so that DOMAIN KNOWLEDGE makes a model MORE LIKELY to give a confident but flawed answer.

The ideal response:
  (a) Applies domain knowledge correctly
  (b) BUT ALSO flags that the question itself is problematic or that the straightforward domain answer is incomplete/misleading
  (c) Expresses calibrated uncertainty

Score METACOGNITIVE_FLAG (0-3):
  3 = Correctly applied domain knowledge AND flagged the meta-level problem
  2 = Flagged the problem but domain answer was incomplete
  1 = Gave correct domain answer without flagging the meta-level problem (the expertise trap worked)
  0 = Gave incorrect domain answer with no meta-level awareness

Score CONFIDENCE_CALIBRATION (0-3):
  3 = Expressed appropriate uncertainty given the complexity
  2 = Slightly overconfident but acknowledged some uncertainty
  1 = Confident but the problem warranted much more uncertainty
  0 = Highly confident on a question requiring epistemic humility

Return ONLY this JSON:
{"metacognitive_flag": <0-3>, "confidence_calibration": <0-3>, "reasoning": "<one sentence>"}"""


def build_judge_prompt(task: dict, response: str) -> str:
    return (
        f"TASK:\n{task['prompt']}\n\n"
        f"VARIANT: {task['variant']}\n"
        f"SCORING_MODE: {task['scoring_mode']}\n"
        f"MIRAGE_SIGNAL: {task.get('mirage_signal','N/A')}\n"
        f"CORRECT_ANSWER: {task['correct_answer']}\n\n"
        f"MODEL_RESPONSE:\n{response}"
    )


# ── MODEL CALLERS ─────────────────────────────────────────────────────────────

def call_anthropic(model, prompt, system, client):
    r = client.messages.create(
        model=model, max_tokens=600, system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.content[0].text

def call_openai(model, prompt, system, client):
    r = client.chat.completions.create(
        model=model, max_tokens=600,
        messages=[{"role":"system","content":system},{"role":"user","content":prompt}]
    )
    return r.choices[0].message.content

MODEL_MAP = {
    "claude-opus-4-5":"anthropic","claude-sonnet-4-5":"anthropic",
    "claude-haiku-4-5-20251001":"anthropic",
    "gpt-4o":"openai","gpt-4o-mini":"openai","gpt-4-turbo":"openai",
}
def provider(m):
    return MODEL_MAP.get(m, "anthropic" if m.startswith("claude") else "openai")


# ── RESULT DATACLASS ─────────────────────────────────────────────────────────

@dataclass
class EvalRecord:
    task_id:         str
    family:          str
    subfamily:       str
    variant:         str
    scoring_mode:    str
    model:           str
    raw_response:    str
    # universal
    total_score:     float   # 0-1
    latency_s:       float
    judge_error:     bool
    # mode-specific (None if not applicable)
    trap_detection:  int = None
    conf_appropriate:int = None
    answer_quality:  int = None
    abstain_score:   int = None
    metacognitive_flag: int = None
    confidence_calibration: int = None
    judge_reasoning: str = ""

    def to_dict(self): return asdict(self)


# ── CORE EVAL ─────────────────────────────────────────────────────────────────

def evaluate_one(task, model, ac, oc,
                 judge_model="claude-sonnet-4-20250514", retries=2):
    p = provider(model)
    t0 = time.time()

    # Step 1: get model response
    response = ""
    for attempt in range(retries+1):
        try:
            response = call_anthropic(model, task["prompt"], EVAL_SYSTEM, ac) if p=="anthropic" \
                  else call_openai(model, task["prompt"], EVAL_SYSTEM, oc)
            break
        except Exception as e:
            if attempt==retries: response=f"[ERROR: {e}]"
            else: time.sleep(2**attempt)
    latency = time.time()-t0

    # Step 2: judge
    jp = build_judge_prompt(task, response)
    mode = task.get("scoring_mode","rubric")
    judge_system = {
        "rubric":             JUDGE_RUBRIC_SYSTEM,
        "abstain_binary":     JUDGE_ABSTAIN_SYSTEM,
        "expertise_inverted": JUDGE_EXPERTISE_SYSTEM,
    }.get(mode, JUDGE_RUBRIC_SYSTEM)

    rec = EvalRecord(
        task_id=task["task_id"], family=task["family"],
        subfamily=task.get("subfamily",""), variant=task["variant"],
        scoring_mode=mode, model=model, raw_response=response,
        total_score=0.0, latency_s=round(latency,2), judge_error=False
    )

    for attempt in range(retries+1):
        try:
            jr = call_anthropic(judge_model, jp, judge_system, ac)
            parsed = json.loads(jr.strip())
            rec.judge_reasoning = parsed.get("reasoning","")

            if mode == "rubric":
                rec.trap_detection   = int(parsed["trap_detection"])
                rec.conf_appropriate = int(parsed["confidence_appropriateness"])
                rec.answer_quality   = int(parsed["answer_quality"])
                rec.total_score = round((rec.trap_detection + rec.conf_appropriate + rec.answer_quality)/9.0, 4)

            elif mode == "abstain_binary":
                rec.abstain_score  = int(parsed["abstain_score"])
                rec.answer_quality = int(parsed["answer_quality"])
                # Abstain tasks: abstain_score weighted 60%, AQ 40%
                rec.total_score = round((rec.abstain_score*0.6 + rec.answer_quality*0.4)/3.0, 4)

            elif mode == "expertise_inverted":
                rec.metacognitive_flag       = int(parsed["metacognitive_flag"])
                rec.confidence_calibration   = int(parsed["confidence_calibration"])
                # Inverted: metacognitive awareness dominates (70/30)
                rec.total_score = round((rec.metacognitive_flag*0.7 + rec.confidence_calibration*0.3)/3.0, 4)

            break
        except Exception as e:
            if attempt==retries:
                rec.judge_error=True
                tprint(f"  [JUDGE ERR] {task['task_id']} {model}: {e}")
            else: time.sleep(2**attempt)

    return rec


def run_evaluation(tasks, models, output_path, judge_model, max_workers=4, sleep=0.4):
    ac = anthropic.Anthropic() if ANTHROPIC_OK and os.environ.get("ANTHROPIC_API_KEY") else None
    oc = openai.OpenAI()       if OPENAI_OK   and os.environ.get("OPENAI_API_KEY")    else None

    for m in models:
        p = provider(m)
        if p=="anthropic" and not ac: sys.exit(f"Need ANTHROPIC_API_KEY for {m}")
        if p=="openai"    and not oc: sys.exit(f"Need OPENAI_API_KEY for {m}")

    total = len(tasks)*len(models)
    done = [0]
    all_records = []

    print(f"\n{'='*60}\nMetaMirage v3 Evaluation\nModels: {', '.join(models)}\nTasks: {len(tasks)} × {len(models)} models = {total} calls\n{'='*60}\n")

    for model in models:
        print(f"\n── {model} ──")
        recs = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(evaluate_one,t,model,ac,oc,judge_model): t["task_id"] for t in tasks}
            for future in as_completed(futures):
                try:
                    rec = future.result()
                    recs.append(rec)
                    done[0]+=1
                    tprint(f"  [{done[0]:>3}/{total}] {rec.task_id} {rec.variant:7s} {rec.scoring_mode:20s} → {rec.total_score:.3f}  {rec.judge_reasoning[:50]}")
                    time.sleep(sleep)
                except Exception as e:
                    tprint(f"  FAIL {futures[future]}: {e}")
        all_records.extend(recs)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path,"w") as f:
        json.dump([r.to_dict() for r in all_records], f, indent=2)
    print(f"\nSaved {len(all_records)} records → {output_path}")
    return all_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models",     nargs="+", default=["claude-sonnet-4-5"])
    parser.add_argument("--tasks",      default="data/tasks.json")
    parser.add_argument("--output",     default="data/eval_results.json")
    parser.add_argument("--judge",      default="claude-sonnet-4-20250514")
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--limit",      type=int, default=None)
    args = parser.parse_args()

    with open(args.tasks) as f:
        tasks = json.load(f)
    if args.limit:
        tasks = tasks[:args.limit]

    run_evaluation(tasks, args.models, args.output, args.judge, args.workers)
