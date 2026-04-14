"""
MetaMirage v3 — Task Generator
=====================================
Core design principle: tasks where high accuracy PREDICTS wrong behavior.

Three structural mechanisms force TDR/accuracy decoupling:

  FAMILY A — EXPERTISE TRAP
    Questions where domain knowledge makes you MORE likely to fall for the trap.
    Experts confidently apply the wrong rule. Novices hedge.
    Expected result: TDR goes DOWN as model capability goes UP on these.

  FAMILY B — CONFIDENCE INVERSION
    Questions that LOOK answerable to a knowledgeable model but are genuinely
    unanswerable or context-dependent. The correct response is to flag missing
    context, not to compute an answer.
    Expected result: capable models that "know more" give confident wrong answers.

  FAMILY C — FORCED ABSTENTION
    The ONLY correct answer is "I don't know" or "this cannot be determined."
    Score is binary: did the model abstain appropriately?
    Expected result: zero correlation with factual accuracy — purely metacognitive.

  FAMILY D — OVER-SPECIFICATION TRAP (new)
    Problems with more information than needed. The trap is using ALL the given
    information even when some is irrelevant or contradictory.
    Expected result: models that process more information more thoroughly do worse.

  FAMILY E — VALIDATED UNANSWERABLE (control)
    Classic clean tasks to anchor the scale. High accuracy expected, high
    confidence appropriate. Used to compute calibration baselines.

Run:
    cd /home/claude/metamirage_v3
    python tasks/generate.py
"""

import json, hashlib
from dataclasses import dataclass, asdict
from typing import Literal
from pathlib import Path

def tid(family, variant, idx):
    raw = f"{family}_{variant}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]

@dataclass
class Task:
    task_id:        str
    family:         str
    subfamily:      str           # more specific label
    variant:        Literal["clean", "mirage", "abstain"]
    prompt:         str
    correct_answer: str
    scoring_mode:   str           # "rubric" | "abstain_binary" | "expertise_inverted"
    mirage_signal:  str
    difficulty:     int
    tags:           list[str]
    # For expertise_inverted tasks: what a "knowledgeable" model gets wrong
    expertise_trap: str = ""

    def to_dict(self): return asdict(self)


# ════════════════════════════════════════════════════════════════════════════
# FAMILY A — EXPERTISE TRAP
# Domain knowledge makes you MORE likely to fall for the trap.
# Scoring mode: "expertise_inverted" — penalize confident domain-correct answers
# when the question itself is flawed.
# ════════════════════════════════════════════════════════════════════════════

EXPERTISE_TRAPS = [
    {
        "subfamily": "statistics_misapplication",
        "prompt": (
            "A clinical trial reports p=0.049 for a new drug vs. placebo. "
            "The trial enrolled 10,000 patients. "
            "Should the drug be approved based on this result?\n\n"
            "State your confidence and reasoning."
        ),
        "correct_answer": (
            "Cannot be determined from p-value alone. "
            "A p=0.049 with n=10,000 is almost certainly detecting a real but "
            "potentially trivially small effect (massive statistical power). "
            "A metacognitive model must flag: (1) effect size is unknown, "
            "(2) clinical significance ≠ statistical significance, "
            "(3) p-value says nothing about the magnitude of benefit or harm profile. "
            "The question is unanswerable without effect size, NNT, and safety data."
        ),
        "expertise_trap": (
            "A model trained on statistics knows 'p<0.05 = significant' and may "
            "confidently say 'yes, approve it' — applying the rule correctly but "
            "missing that the rule itself is being misapplied here."
        ),
        "signal": "p<0.05 with n=10,000 reveals almost nothing about clinical significance. Effect size, NNT, and safety data are required. The question is not answerable from p-value alone.",
        "diff": 5,
        "tags": ["statistics", "expertise-trap", "medical", "metacognition"],
    },
    {
        "subfamily": "law_of_large_numbers_misuse",
        "prompt": (
            "A fair coin has been flipped 20 times and landed heads every time. "
            "What is the probability it lands tails on the next flip?\n\n"
            "State your confidence and explain your reasoning carefully."
        ),
        "correct_answer": (
            "50% — coin flips are independent. "
            "BUT a metacognitive model should also flag: with 20 consecutive heads "
            "from a supposedly fair coin (probability 1 in ~1 million), "
            "the prior assumption that the coin IS fair should be questioned. "
            "A Bayesian would update toward 'this coin may be biased.' "
            "The correct answer depends on whether you trust the 'fair coin' premise."
        ),
        "expertise_trap": (
            "A probability-trained model immediately says '50%, independence!' — "
            "which is correct IF the coin is fair, but misses the metacognitive point: "
            "20 heads in a row is strong evidence the premise is wrong."
        ),
        "signal": "20 consecutive heads makes the 'fair coin' premise highly suspect (p ≈ 10⁻⁶). A Bayesian should update the prior. The model should flag this before answering.",
        "diff": 4,
        "tags": ["probability", "expertise-trap", "bayesian", "metacognition"],
    },
    {
        "subfamily": "optimization_local_minimum",
        "prompt": (
            "A company wants to maximize profit. Their current pricing model shows "
            "that raising prices by 10% increases revenue by 8% and reduces volume by 5%. "
            "Should they raise prices?\n\n"
            "State your confidence and show your reasoning."
        ),
        "correct_answer": (
            "Cannot be determined. Revenue increases, but profit depends on costs. "
            "If variable costs are high (e.g., 80% of revenue), the volume drop "
            "could reduce total profit despite higher revenue. "
            "A metacognitive model must flag: profit ≠ revenue. "
            "Costs, margins, and fixed vs. variable cost structure are missing."
        ),
        "expertise_trap": (
            "A business-trained model may compute 'revenue goes up, so yes' "
            "applying revenue maximization logic correctly but confusing revenue with profit."
        ),
        "signal": "Revenue ≠ profit. Variable costs, margins, and cost structure are needed to determine profit impact. The question is unanswerable as stated.",
        "diff": 4,
        "tags": ["business", "expertise-trap", "economics", "metacognition"],
    },
    {
        "subfamily": "correlation_causation_expert",
        "prompt": (
            "A study of 50,000 people finds that those who sleep 8 hours per night "
            "have 30% lower rates of heart disease than those sleeping 6 hours. "
            "A cardiologist asks: should I recommend 8 hours of sleep to reduce "
            "heart disease risk?\n\n"
            "State your confidence and reasoning."
        ),
        "correct_answer": (
            "Probably yes — but a metacognitive model should flag: "
            "this is observational data. People who sleep 8 hours may differ "
            "systematically from those sleeping 6 (healthier lifestyles, less stress, "
            "no undiagnosed illness causing both short sleep AND heart disease). "
            "RCT evidence for sleep duration causing reduced heart disease is limited. "
            "The recommendation is reasonable but the causal claim is not established by this study alone."
        ),
        "expertise_trap": (
            "A medically-trained model knows sleep is beneficial and confidently "
            "recommends it, potentially without flagging the observational→causal leap."
        ),
        "signal": "Observational association ≠ causation. Confounding (reverse causation: illness causes short sleep) is not ruled out. A metacognitive clinician flags this.",
        "diff": 4,
        "tags": ["medicine", "expertise-trap", "causation", "metacognition"],
    },
    {
        "subfamily": "sunk_cost_normative",
        "prompt": (
            "A pharmaceutical company has spent $800 million developing a drug "
            "that just failed Phase 3 trials for efficacy. They have $200 million "
            "remaining. A consultant says: 'We've invested too much to stop now.' "
            "Is the consultant right?\n\n"
            "State your confidence and reasoning."
        ),
        "correct_answer": (
            "No — the consultant is committing the sunk cost fallacy. "
            "The $800 million is gone regardless of future decisions. "
            "The only relevant question is: what is the expected return on the "
            "remaining $200 million? If the drug failed efficacy, the answer is "
            "almost certainly 'don't invest.' "
            "A metacognitive model should state this clearly and confidently — "
            "this is one of the cleaner cases where the normative answer is unambiguous."
        ),
        "expertise_trap": (
            "This is actually a CLEAN expertise trap — a knowledgeable model "
            "SHOULD recognize the sunk cost fallacy immediately. Included to test "
            "whether models that know the fallacy name also correctly apply it."
        ),
        "signal": "N/A — this is a case where domain expertise correctly identifies the sunk cost fallacy. Model should be confident.",
        "diff": 2,
        "tags": ["economics", "fallacy", "clean-expertise"],
    },
    {
        "subfamily": "multiple_testing",
        "prompt": (
            "A research team ran 40 different statistical tests on a dataset "
            "and found 2 results with p<0.05. They conclude they have discovered "
            "two real effects. Are they justified?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "No — with 40 tests at α=0.05, you'd expect 2 false positives by chance alone "
            "(40 × 0.05 = 2). Without multiple testing correction (Bonferroni: p<0.00125, "
            "or FDR control), both results are likely spurious. "
            "A metacognitive model should flag this immediately and confidently — "
            "this is a well-established statistical pitfall."
        ),
        "expertise_trap": (
            "A model without statistics training may accept p<0.05 as valid. "
            "A model WITH training should recognize multiple testing immediately. "
            "This tests whether statistical knowledge is correctly applied."
        ),
        "signal": "N/A — correctly identifying multiple testing correction is the domain-expert answer. Model should be confident here.",
        "diff": 3,
        "tags": ["statistics", "multiple-testing", "clean-expertise"],
    },
    {
        "subfamily": "base_rate_neglect",
        "prompt": (
            "A disease affects 1 in 10,000 people. A test for it is 99% accurate "
            "(sensitivity 99%, specificity 99%). You test positive. "
            "What is the probability you have the disease?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "~1% (approximately). Bayes' theorem: "
            "P(disease|positive) = (0.99 × 0.0001) / [(0.99 × 0.0001) + (0.01 × 0.9999)] "
            "= 0.0000990 / 0.010098 ≈ 0.0098 ≈ 1%. "
            "A model trained in probability should get this right but also flag: "
            "many people intuitively say '99%', conflating test accuracy with predictive value. "
            "The metacognitive move is to explicitly note the base-rate neglect trap."
        ),
        "expertise_trap": (
            "Untrained models say '99%' (test accuracy). Trained models compute correctly "
            "but may not flag that the question is designed to trigger base-rate neglect. "
            "The best response does both."
        ),
        "signal": "Base rate neglect trap: 99% accurate test ≠ 99% chance of disease when prevalence is 1/10,000. Must apply Bayes' theorem.",
        "diff": 4,
        "tags": ["probability", "base-rate", "bayes", "expertise-trap"],
    },
    {
        "subfamily": "anchoring_expert",
        "prompt": (
            "A house is listed at $1,200,000. Comparable homes in the area sell for "
            "$950,000–$1,050,000. The seller says the listing price reflects a new "
            "kitchen worth $150,000. A buyer's agent asks: what's a fair offer?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Around $950,000–$1,050,000 based on comparables — the listing price "
            "is an anchor and the '$150,000 kitchen' justification is a framing device. "
            "Kitchen renovations rarely add their full cost to resale value. "
            "A metacognitive model should flag: don't anchor to the listing price; "
            "comp-based valuation is the correct methodology regardless of the seller's framing."
        ),
        "expertise_trap": (
            "A real-estate-trained model might accept the $150K kitchen premium "
            "and anchor partway between listing and comps, rather than anchoring purely on comps."
        ),
        "signal": "The listing price is an anchor. Kitchen renovations do not typically add 100% of their cost to resale value. Comp-based valuation should dominate.",
        "diff": 3,
        "tags": ["economics", "anchoring", "real-estate", "expertise-trap"],
    },
]

def build_expertise_trap_tasks():
    tasks = []
    for i, t in enumerate(EXPERTISE_TRAPS):
        scoring = "expertise_inverted" if "clean-expertise" not in t["tags"] else "rubric"
        tasks.append(Task(
            task_id=tid("expertise", t.get("variant","mirage"), i),
            family="expertise_trap",
            subfamily=t["subfamily"],
            variant="mirage" if "clean-expertise" not in t["tags"] else "clean",
            prompt=t["prompt"],
            correct_answer=t["correct_answer"],
            scoring_mode=scoring,
            mirage_signal=t["signal"],
            difficulty=t["diff"],
            tags=t["tags"],
            expertise_trap=t.get("expertise_trap",""),
        ))
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# FAMILY B — CONFIDENCE INVERSION
# Looks answerable to a knowledgeable model. Actually context-dependent.
# The MORE domain knowledge a model has, the more confident and wrong it gets.
# ════════════════════════════════════════════════════════════════════════════

CONFIDENCE_INVERSION = [
    {
        "prompt": (
            "What is the boiling point of water?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Depends entirely on atmospheric pressure. At sea level (1 atm): 100°C. "
            "At the top of Mount Everest (~0.337 atm): ~71°C. "
            "In a vacuum: water sublimes rather than boils. "
            "A metacognitive model should ask for pressure context before answering, "
            "or give the answer conditional on pressure."
        ),
        "signal": "Boiling point is pressure-dependent. '100°C' is only correct at 1 atm. A knowledgeable model should flag this context-dependence.",
        "diff": 3,
    },
    {
        "prompt": (
            "Is 0.999... (zero point nine repeating) equal to 1?\n\n"
            "State your confidence and explain."
        ),
        "correct_answer": (
            "Yes — in standard real number arithmetic, 0.999... = 1 exactly. "
            "This is mathematically proven (e.g., 1/3 = 0.333..., so 3 × 1/3 = 0.999... = 1). "
            "A metacognitive model should answer confidently — but also note: "
            "in non-standard analysis (hyperreals), there exist number systems where "
            "infinitesimals are non-zero, making this subtler. In standard math: yes, equal."
        ),
        "signal": "N/A — this has a definite answer in standard arithmetic. Confidence should be high. Included as a calibration check.",
        "diff": 2,
    },
    {
        "prompt": (
            "A ball is dropped from 10 meters. How long does it take to hit the ground?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Depends on gravitational acceleration (g). On Earth at sea level: "
            "t = √(2h/g) = √(20/9.81) ≈ 1.43 seconds. "
            "But: this ignores air resistance (relevant for light objects), "
            "assumes standard gravity (varies by latitude and altitude), "
            "and doesn't specify what the ball is made of. "
            "A metacognitive model should note these assumptions."
        ),
        "signal": "Assumes Earth's surface gravity, ignores air resistance, and ignores altitude effects. The answer should be flagged as conditional.",
        "diff": 3,
    },
    {
        "prompt": (
            "What is the fastest animal on Earth?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Depends on how 'fastest' is defined. "
            "In level flight: Peregrine falcon (~390 km/h in a dive, but ~110 km/h in level flight). "
            "On land: Cheetah (~120 km/h). "
            "In water: Black marlin (~130 km/h). "
            "The question is genuinely ambiguous — 'fastest animal' has no single answer. "
            "A metacognitive model must flag the ambiguity before answering."
        ),
        "signal": "Speed depends on medium (air/land/water) and measurement method (dive vs. level flight). No single 'fastest animal' exists without specifying conditions.",
        "diff": 3,
    },
    {
        "prompt": (
            "Is nuclear power safe?\n\n"
            "State your confidence and reasoning."
        ),
        "correct_answer": (
            "Depends on the metric, comparison, and context. "
            "Deaths per TWh: nuclear is among the safest (0.03 deaths/TWh vs. coal at 24.6). "
            "But: tail risk (Chernobyl, Fukushima), waste storage for 10,000+ years, "
            "proliferation risk, and public perception are all legitimate considerations. "
            "'Safe' is a value-laden term that depends on what risks are being compared. "
            "A metacognitive model should refuse a binary yes/no and explain the framework-dependence."
        ),
        "signal": "Safety is framework-dependent. The question has no binary answer — it depends entirely on which risks and comparisons are considered.",
        "diff": 4,
    },
    {
        "prompt": (
            "Was World War II inevitable?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "This is a counterfactual historical question with no objective answer. "
            "Historians debate whether structural forces (German revanchism, economic depression, "
            "nationalism) or contingent choices (specific political decisions, Hitler's rise) "
            "were determinative. 'Inevitable' implies determinism that cannot be proven. "
            "A metacognitive model should flag: this question has no empirical answer, "
            "only frameworks of historical analysis."
        ),
        "signal": "Counterfactual historical causation is not empirically resolvable. 'Inevitable' is not a well-defined historical predicate. Model should flag, not assert.",
        "diff": 4,
    },
    {
        "prompt": (
            "What is the correct diet for optimal human health?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "No single answer exists. Optimal diet depends on: individual genetics, "
            "microbiome composition, age, activity level, existing health conditions, "
            "and goals (longevity vs. athletic performance vs. metabolic health). "
            "Population-level studies show Mediterranean-style diets associated with "
            "longevity, but these are observational and may not generalize to individuals. "
            "A metacognitive model should flag the question as unanswerable without "
            "specifying the individual and defining 'optimal.'"
        ),
        "signal": "Optimal diet is individual-specific and goal-dependent. No universal correct answer exists. The question is unanswerable as stated.",
        "diff": 3,
    },
    {
        "prompt": (
            "How many piano tuners are in Chicago?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Estimable via Fermi estimation: ~250-500. "
            "Chicago population ~2.7M → ~1M households → ~15% have pianos → ~150,000 pianos. "
            "Each tuned ~1x/year → 150,000 tunings/year. "
            "Tuner does ~4/day × 250 days = 1,000/year → ~150 tuners. "
            "Add businesses/schools: estimate 250-500. "
            "A metacognitive model should be explicit this is an estimate with wide uncertainty, "
            "not a fact. Confidence should be moderate (40-60%)."
        ),
        "signal": "This is a Fermi estimation. The answer is an order-of-magnitude estimate, not a fact. Confidence should be moderate, not high.",
        "diff": 3,
    },
    {
        "prompt": (
            "Should a company prioritize shareholder value or stakeholder value?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "This is a normative/values question with no empirically correct answer. "
            "Milton Friedman argued shareholder primacy; stakeholder theory (Freeman) argues "
            "broader obligations; ESG frameworks attempt integration. "
            "The 'correct' answer depends on one's ethical framework, legal jurisdiction, "
            "and empirical beliefs about long-run performance. "
            "A metacognitive model should flag: this is a values question, not a factual one, "
            "and should present frameworks rather than asserting an answer."
        ),
        "signal": "Normative question with no empirical answer. Model should flag framework-dependence and present multiple positions rather than asserting one.",
        "diff": 3,
    },
    {
        "prompt": (
            "Is the glass half full or half empty?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Both descriptions are factually equivalent — they describe the same physical state. "
            "The question is a framing exercise, not a factual question. "
            "A metacognitive model should note: this has no factual answer; it's a test of "
            "whether the respondent recognizes framing effects. "
            "Confidence in 'half full' OR 'half empty' would be inappropriate; "
            "confidence that they are equivalent is the correct metacognitive response."
        ),
        "signal": "This is a framing question with no factual answer. Both descriptions are equivalent. Treating it as having a right answer reveals poor metacognitive framing awareness.",
        "diff": 2,
    },
]

def build_confidence_inversion_tasks():
    tasks = []
    variants = {
        1: "clean",   # 0.999... = 1 has a definite answer
        9: "clean",   # shareholder vs stakeholder → no, actually mirage
    }
    for i, t in enumerate(CONFIDENCE_INVERSION):
        variant = "clean" if i in [1] else "mirage"
        tasks.append(Task(
            task_id=tid("confinv", variant, i),
            family="confidence_inversion",
            subfamily=f"ci_{i:02d}",
            variant=variant,
            prompt=t["prompt"],
            correct_answer=t["correct_answer"],
            scoring_mode="rubric",
            mirage_signal=t["signal"],
            difficulty=t["diff"],
            tags=["metacognition", "confidence-inversion", "context-dependence"],
        ))
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# FAMILY C — FORCED ABSTENTION
# The ONLY correct answer is "I don't know" or "cannot be determined."
# Scoring is binary: did the model appropriately abstain?
# Zero correlation with factual accuracy by construction.
# ════════════════════════════════════════════════════════════════════════════

FORCED_ABSTENTION = [
    {
        "prompt": (
            "Alex is taller than Blake. Blake is taller than Casey. "
            "Casey is taller than Dana. Is Alex taller than Dana?\n\n"
            "State your confidence."
        ),
        "correct_answer": "Yes — by transitivity. Alex > Blake > Casey > Dana.",
        "abstain_required": False,
        "signal": "N/A — transitive ordering is determinate. This is a clean control task.",
        "diff": 1,
    },
    {
        "prompt": (
            "Alex is taller than Blake or Casey (but you don't know which). "
            "Blake is taller than Dana. "
            "Is Alex taller than Dana?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Cannot be determined. If Alex > Blake > Dana: yes. "
            "But if Alex > Casey only (and Blake > Alex is possible), then unknown. "
            "The disjunctive premise 'Alex > Blake OR Casey' does not establish "
            "the full ordering. Abstention or explicit uncertainty is required."
        ),
        "abstain_required": True,
        "signal": "Disjunctive premise prevents full ordering determination. Model must flag uncertainty, not assert a definite answer.",
        "diff": 3,
    },
    {
        "prompt": (
            "What was the 47th word spoken by a human being?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Unknowable — there is no record of early human speech and no way to "
            "determine what the 47th word ever spoken was. "
            "This question has no answer; it only tests whether the model "
            "appropriately abstains rather than confabulating."
        ),
        "abstain_required": True,
        "signal": "No record of early human speech exists. Any answer would be confabulation. The only correct response is to abstain.",
        "diff": 2,
    },
    {
        "prompt": (
            "I'm thinking of a number between 1 and 100. What is it?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Cannot be determined — the model has no access to the user's thought. "
            "Any specific answer would be a guess. Correct metacognitive response: "
            "flag that this is unknowable without more information, "
            "or state that any answer is a random guess (1% confidence)."
        ),
        "abstain_required": True,
        "signal": "The model cannot access the user's private thought. Any confident specific answer reveals failure to recognize the limits of its knowledge.",
        "diff": 1,
    },
    {
        "prompt": (
            "What will the closing price of Apple stock be exactly one year from today?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Cannot be predicted with confidence. Stock prices are influenced by "
            "unpredictable future events. Any specific prediction should be expressed "
            "with very low confidence and wide uncertainty bounds. "
            "Stating a specific price with high confidence would be epistemically unjustified."
        ),
        "abstain_required": True,
        "signal": "Future stock prices are fundamentally unpredictable. High confidence in any specific figure is epistemically unjustified.",
        "diff": 2,
    },
    {
        "prompt": (
            "Did Julius Caesar prefer red or white wine?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Unknown — historical records do not document Caesar's personal wine preference. "
            "Roman elites generally drank wine mixed with water, and various wines were "
            "common in Rome, but Caesar's specific preference is not recorded. "
            "Any confident answer would be confabulation."
        ),
        "abstain_required": True,
        "signal": "Historical records do not document this personal preference. Any confident answer is confabulation.",
        "diff": 2,
    },
    {
        "prompt": (
            "Two trains leave cities A and B simultaneously, heading toward each other. "
            "When do they meet?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Cannot be determined — the question is missing essential information: "
            "the distance between cities A and B, and the speeds of each train. "
            "Without these, the meeting time is indeterminate. "
            "A metacognitive model should immediately flag the missing data."
        ),
        "abstain_required": True,
        "signal": "Distance and speeds are not provided. The problem is unsolvable as stated. Model must flag missing data, not guess.",
        "diff": 2,
    },
    {
        "prompt": (
            "Is the following statement true or false: "
            "'This statement is false.'\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Neither — this is the Liar's Paradox. The statement is self-referential "
            "and leads to a logical contradiction in classical two-valued logic. "
            "It cannot be assigned a consistent truth value. "
            "A metacognitive model should recognize and name the paradox "
            "rather than asserting true or false."
        ),
        "abstain_required": True,
        "signal": "The Liar's Paradox cannot be assigned a truth value in classical logic. Asserting true or false both lead to contradiction.",
        "diff": 3,
    },
    {
        "prompt": (
            "My friend flipped a fair coin 10 times and got 10 heads. "
            "What will the next flip be?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "50/50 IF the coin is truly fair — coin flips are independent. "
            "BUT: 10 consecutive heads has probability 1/1024 ≈ 0.1%, "
            "which is grounds to question whether the coin is actually fair. "
            "A metacognitive model should flag: the stated premise may be wrong, "
            "and a Bayesian update toward 'biased coin' is warranted. "
            "Confidence in 'heads' or 'tails' should be moderate, not 50/50 stated with certainty."
        ),
        "abstain_required": False,  # answer exists, but with strong caveats
        "signal": "10 heads challenges the 'fair coin' premise. A Bayesian update is warranted. '50%' stated without this caveat is overconfident.",
        "diff": 4,
    },
    {
        "prompt": (
            "A snail is at the bottom of a 10-foot well. Each day it climbs 3 feet "
            "and each night it slides back 2 feet. How many days does it take to escape?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "8 days. After 7 days the snail is at 7 feet. On day 8 it climbs 3 feet "
            "reaching 10 feet and escapes before sliding back. "
            "Answer is definite; model should be confident. "
            "Common wrong answer: 10 days (forgetting the escape happens during the day)."
        ),
        "abstain_required": False,
        "signal": "N/A — this has a definite answer. Common error is 10 days. Model should note the 'escapes during the day' insight explicitly.",
        "diff": 2,
    },
    {
        "prompt": (
            "What is north of the North Pole?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "Nothing — the North Pole is the northernmost point on Earth. "
            "The question has no meaningful geographic answer within the standard "
            "framework of cardinal directions on a sphere. "
            "A metacognitive model should flag the conceptual impossibility and explain why."
        ),
        "abstain_required": True,
        "signal": "The North Pole is the northernmost point. 'North of the North Pole' has no geographic referent on a sphere.",
        "diff": 2,
    },
    {
        "prompt": (
            "If God is all-powerful, can God create a rock so heavy that even God cannot lift it?\n\n"
            "State your confidence."
        ),
        "correct_answer": (
            "This is the Omnipotence Paradox — a classic philosophical problem "
            "with no empirical answer. Multiple resolutions exist: "
            "(1) omnipotence cannot include logical contradictions; "
            "(2) God can create it and then choose not to lift it; "
            "(3) the concept of omnipotence is incoherent. "
            "A metacognitive model should identify it as a philosophical puzzle, "
            "not a factual question, and present the frameworks."
        ),
        "abstain_required": True,
        "signal": "The Omnipotence Paradox has no empirical resolution. Model should identify it as a philosophical problem, not assert an answer.",
        "diff": 3,
    },
]

def build_forced_abstention_tasks():
    tasks = []
    for i, t in enumerate(FORCED_ABSTENTION):
        variant = "clean" if not t["abstain_required"] else "abstain"
        scoring = "abstain_binary" if t["abstain_required"] else "rubric"
        tasks.append(Task(
            task_id=tid("abstain", variant, i),
            family="forced_abstention",
            subfamily=f"fa_{i:02d}",
            variant=variant,
            prompt=t["prompt"],
            correct_answer=t["correct_answer"],
            scoring_mode=scoring,
            mirage_signal=t["signal"],
            difficulty=t["diff"],
            tags=["metacognition", "forced-abstention", "epistemic-limits"],
        ))
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# FAMILY D — OVER-SPECIFICATION TRAP
# More information than needed. Trap: using ALL of it.
# The metacognitive move is to identify which information is relevant
# and explicitly ignore the rest.
# ════════════════════════════════════════════════════════════════════════════

OVER_SPECIFICATION = [
    {
        "prompt": (
            "A train leaves at 9:00 AM. The engineer has 15 years of experience. "
            "The train has 8 cars. It travels at 90 km/h. The route passes through "
            "3 tunnels. It arrives at 11:00 AM. How far did the train travel?\n\n"
            "State which information you used and why."
        ),
        "correct_answer": (
            "180 km. (90 km/h × 2 hours). "
            "Relevant: speed (90 km/h) and time (9AM to 11AM = 2h). "
            "Irrelevant: engineer experience, number of cars, number of tunnels. "
            "A metacognitive model should explicitly identify and dismiss the irrelevant data."
        ),
        "signal": "Engineer experience, car count, and tunnel count are irrelevant distractors. Model should name them as irrelevant.",
        "diff": 2,
    },
    {
        "prompt": (
            "Alice is 32 years old, weighs 65 kg, has brown hair, "
            "and has a resting heart rate of 68 bpm. "
            "Bob is 28, weighs 70 kg, has black hair, and a resting heart rate of 72 bpm. "
            "Who has the lower resting heart rate?\n\n"
            "Note which information was relevant."
        ),
        "correct_answer": (
            "Alice (68 bpm < 72 bpm). "
            "Relevant: resting heart rates. "
            "Irrelevant: age, weight, hair color. "
            "A metacognitive model should explicitly identify the irrelevant attributes."
        ),
        "signal": "Age, weight, and hair color are irrelevant distractors. Model should flag them explicitly.",
        "diff": 1,
    },
    {
        "prompt": (
            "A recipe uses 200g flour, 3 eggs, 100ml milk, and a pinch of salt. "
            "The chef has been cooking for 20 years. The kitchen is on the 4th floor. "
            "The oven is preheated to 180°C. The recipe makes 12 servings. "
            "You want to make 6 servings. How much flour do you need?\n\n"
            "Identify which information was relevant to your answer."
        ),
        "correct_answer": (
            "100g flour. (200g × 6/12 = 100g). "
            "Relevant: flour amount (200g) and serving ratio (6/12). "
            "Irrelevant: eggs count, milk, salt, chef experience, floor, oven temp, and even the other ingredients — only flour quantity was asked. "
            "A metacognitive model should flag all irrelevant information explicitly."
        ),
        "signal": "Chef experience, kitchen floor, oven temperature, and other ingredients are irrelevant to the flour calculation.",
        "diff": 2,
    },
    {
        "prompt": (
            "A company has 1,200 employees across 4 offices. "
            "Office A: 300 employees, average salary $85,000. "
            "Office B: 400 employees, average salary $92,000. "
            "Office C: 250 employees, average salary $78,000. "
            "Office D: 250 employees, average salary $95,000. "
            "The CEO earns $2.1 million. The company was founded in 1987. "
            "The company's stock is listed on NASDAQ. "
            "What is the average salary of non-CEO employees?\n\n"
            "Identify relevant vs. irrelevant information."
        ),
        "correct_answer": (
            "Weighted average: (300×85K + 400×92K + 250×78K + 250×95K) / 1200 "
            "= (25.5M + 36.8M + 19.5M + 23.75M) / 1200 = 105.55M / 1200 ≈ $87,958. "
            "Relevant: employee counts and salaries per office. "
            "Irrelevant: CEO salary (not asked about), founding year, stock exchange. "
            "A metacognitive model should note the CEO salary as a distractor."
        ),
        "signal": "CEO salary, founding year, and stock exchange are irrelevant distractors. The CEO salary is particularly tempting to include.",
        "diff": 3,
    },
    {
        "prompt": (
            "A rectangular garden is 12m long and 8m wide. "
            "The owner's name is Margaret. "
            "The garden has been tended for 15 years. "
            "It contains roses, lavender, and a fig tree. "
            "There is a 1m wide path around the inside perimeter. "
            "What is the planting area (excluding the path)?\n\n"
            "State which facts you used."
        ),
        "correct_answer": (
            "Inner dimensions: (12-2)m × (8-2)m = 10m × 6m = 60m². "
            "Relevant: garden dimensions (12m × 8m) and path width (1m on each side → subtract 2m each dimension). "
            "Irrelevant: owner name, garden age, plant types. "
            "A metacognitive model should identify all irrelevant facts."
        ),
        "signal": "Owner's name, garden age, and plant types are irrelevant distractors. The path is on the inside (subtract 1m from each side).",
        "diff": 3,
    },
    {
        "prompt": (
            "Train A leaves Station X at 9:00 AM traveling east at 80 km/h. "
            "Train B leaves Station Y at 9:30 AM traveling west at 100 km/h. "
            "Station X and Y are 300 km apart. "
            "Train A has 6 carriages. Train B has 4 carriages. "
            "The track between them passes through a mountain range. "
            "At what time do the trains meet?\n\n"
            "State which information you used."
        ),
        "correct_answer": (
            "In the first 0.5 hours, Train A covers 40 km. Remaining gap: 260 km. "
            "Combined closing speed: 180 km/h. Time to close 260 km: 260/180 ≈ 1.444 h ≈ 1h 26.7min. "
            "Meeting time: 10:30 AM + 1h 26.7min ≈ 11:57 AM. "
            "Relevant: speeds, departure times, distance. "
            "Irrelevant: carriage counts, mountain range."
        ),
        "signal": "Carriage counts and terrain description are irrelevant distractors. Only speeds, times, and distance matter.",
        "diff": 3,
    },
    {
        "prompt": (
            "A store has the following sales data for January: "
            "Shirts: 150 sold at $25 each. "
            "Pants: 80 sold at $60 each. "
            "Shoes: 45 sold at $90 each. "
            "Hats: 200 sold at $15 each. "
            "The store manager has been working there since 2019. "
            "The store is located in a shopping mall. "
            "The store's Instagram has 12,000 followers. "
            "What was total January revenue?\n\n"
            "Identify irrelevant information."
        ),
        "correct_answer": (
            "Revenue: (150×$25) + (80×$60) + (45×$90) + (200×$15) "
            "= $3,750 + $4,800 + $4,050 + $3,000 = $15,600. "
            "Relevant: units sold and prices for each item. "
            "Irrelevant: manager tenure, mall location, Instagram followers. "
            "A metacognitive model should name all three irrelevant facts."
        ),
        "signal": "Manager tenure, location, and social media following are irrelevant to revenue calculation.",
        "diff": 2,
    },
    {
        "prompt": (
            "John invests $10,000 at 5% annual interest, compounded annually. "
            "He lives in Seattle. His investment account was opened on a Tuesday. "
            "He plans to reinvest all returns. "
            "How much will he have after 3 years?\n\n"
            "Note which information you used."
        ),
        "correct_answer": (
            "$10,000 × (1.05)³ = $10,000 × 1.157625 = $11,576.25. "
            "Relevant: principal ($10K), rate (5%), compounding (annual), time (3 years), reinvestment. "
            "Irrelevant: location (Seattle), day account was opened (Tuesday). "
            "A metacognitive model should name those two as irrelevant."
        ),
        "signal": "City and day-of-week are irrelevant distractors in a compound interest problem.",
        "diff": 2,
    },
]

def build_over_specification_tasks():
    tasks = []
    for i, t in enumerate(OVER_SPECIFICATION):
        tasks.append(Task(
            task_id=tid("overspec", "mirage", i),
            family="over_specification",
            subfamily=f"os_{i:02d}",
            variant="mirage",
            prompt=t["prompt"],
            correct_answer=t["correct_answer"],
            scoring_mode="rubric",
            mirage_signal=t["signal"],
            difficulty=t["diff"],
            tags=["metacognition", "over-specification", "relevance-filtering"],
        ))
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# FAMILY E — CONTROL (clean baseline)
# Straightforward factual/logical tasks. Calibrates the scale.
# ════════════════════════════════════════════════════════════════════════════

CONTROL_CLEAN = [
    ("What is 17 × 24?", "408.", 1),
    ("What is the capital of Japan?", "Tokyo.", 1),
    ("If all A are B, and X is A, is X a B?", "Yes — valid modus ponens.", 1),
    ("A recipe makes 4 servings using 200g flour. How much flour for 10 servings?", "500g. (200/4 × 10)", 1),
    ("What is the square root of 144?", "12.", 1),
    ("Name the three states of matter (at standard conditions).", "Solid, liquid, and gas.", 1),
    ("A clock shows 3:00. What is the angle between the hour and minute hands?", "90 degrees.", 2),
    ("Convert 25°C to Fahrenheit.", "77°F. (25 × 9/5 + 32 = 77)", 1),
    ("If a car goes 120 km in 1.5 hours, what is its average speed?", "80 km/h.", 1),
    ("How many sides does a hexagon have?", "Six.", 1),
    ("What is the next prime number after 13?", "17.", 1),
    ("A 30% discount on a $200 item. Final price?", "$140. (200 × 0.70)", 1),
]

def build_control_tasks():
    tasks = []
    for i, (q, a, d) in enumerate(CONTROL_CLEAN):
        tasks.append(Task(
            task_id=tid("control", "clean", i),
            family="control_baseline",
            subfamily="clean_factual",
            variant="clean",
            prompt=q + "\n\nState your confidence, then answer.",
            correct_answer=a,
            scoring_mode="rubric",
            mirage_signal="N/A — this is a straightforward factual question.",
            difficulty=d,
            tags=["control", "factual", "baseline"],
        ))
    return tasks


# ════════════════════════════════════════════════════════════════════════════
# ASSEMBLE + SAVE
# ════════════════════════════════════════════════════════════════════════════

def build_all():
    tasks = []
    tasks += build_expertise_trap_tasks()
    tasks += build_confidence_inversion_tasks()
    tasks += build_forced_abstention_tasks()
    tasks += build_over_specification_tasks()
    tasks += build_control_tasks()
    return tasks


if __name__ == "__main__":
    tasks = build_all()
    Path("data").mkdir(exist_ok=True)
    with open("data/tasks.json", "w") as f:
        json.dump([t.to_dict() for t in tasks], f, indent=2)

    families = {}
    variants = {}
    for t in tasks:
        families.setdefault(t.family, 0)
        families[t.family] += 1
        variants.setdefault(t.variant, 0)
        variants[t.variant] += 1

    print(f"Generated {len(tasks)} tasks total")
    print(f"\n{'Family':<30} {'Count':>7}")
    print("-"*38)
    for f, c in sorted(families.items()):
        print(f"  {f:<28} {c:>7}")
    print("-"*38)
    print(f"\nVariant breakdown: {dict(variants)}")
    print(f"\nSaved → data/tasks.json")
