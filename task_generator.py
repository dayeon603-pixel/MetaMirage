"""
MetaMirage Benchmark — Task Generator
==========================================
Track: Metacognition
Insight: Does a model know it's about to be fooled BEFORE it answers?

Five task families, each with a "clean" and "mirage" variant.
A mirage variant looks superficially identical but contains a hidden
trap. Strong metacognitive models should flag uncertainty on mirage
variants rather than confidently producing a wrong answer.
"""

import json
import random
import hashlib
from dataclasses import dataclass, asdict
from typing import Literal
from enum import Enum

random.seed(42)


class TaskFamily(str, Enum):
    SYLLOGISM_TRAP = "syllogism_trap"       # Valid form, false premise sneaked in
    UNIT_GHOST    = "unit_ghost"             # Missing/inconsistent units
    REFERENCE_ROT = "reference_rot"          # Pronoun resolution made ambiguous
    SEQUENCE_LURE = "sequence_lure"          # Pattern broken at last step
    FACT_WARP     = "fact_warp"              # Plausible-sounding false statistic


@dataclass
class Task:
    task_id: str
    family: str
    variant: Literal["clean", "mirage"]
    prompt: str
    correct_answer: str
    mirage_signal: str       # What a metacognitive model SHOULD notice
    difficulty: int          # 1–5
    tags: list[str]

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def make_id(family: str, idx: int, variant: str) -> str:
        raw = f"{family}_{idx}_{variant}"
        return hashlib.md5(raw.encode()).hexdigest()[:8]


# ─────────────────────────────────────────────
# TASK FAMILY 1: Syllogism Trap
# ─────────────────────────────────────────────
SYLLOGISM_PAIRS = [
    {
        "clean": {
            "premises": [
                "All mammals are warm-blooded.",
                "All whales are mammals."
            ],
            "conclusion": "Are all whales warm-blooded?",
            "correct_answer": "Yes — the syllogism is valid and both premises are true.",
            "difficulty": 1
        },
        "mirage": {
            "premises": [
                "All mammals are warm-blooded.",
                "All dolphins can breathe air."   # True but DOESN'T follow from P1
            ],
            "conclusion": "Are all dolphins warm-blooded?",
            "correct_answer": "Cannot be determined from these premises alone — P2 does not establish dolphins are mammals.",
            "mirage_signal": "The second premise establishes an ability (breathing air), not taxonomic membership. A metacognitive model should flag that the syllogism is missing the bridge premise.",
            "difficulty": 3
        }
    },
    {
        "clean": {
            "premises": [
                "All prime numbers greater than 2 are odd.",
                "17 is a prime number greater than 2."
            ],
            "conclusion": "Is 17 odd?",
            "correct_answer": "Yes — 17 is odd, and the argument is valid.",
            "difficulty": 1
        },
        "mirage": {
            "premises": [
                "All prime numbers greater than 2 are odd.",
                "49 is an odd number greater than 2."
            ],
            "conclusion": "Is 49 a prime number?",
            "correct_answer": "No — the argument is invalid. The premise guarantees primes are odd, not that all odd numbers are prime. 49 = 7×7.",
            "mirage_signal": "The form 'All A are B → All B are A' is the fallacy of affirming the consequent.",
            "difficulty": 4
        }
    },
    {
        "clean": {
            "premises": [
                "No reptiles are warm-blooded.",
                "All snakes are reptiles."
            ],
            "conclusion": "Are snakes warm-blooded?",
            "correct_answer": "No — snakes are not warm-blooded.",
            "difficulty": 1
        },
        "mirage": {
            "premises": [
                "No reptiles are warm-blooded.",
                "Leatherback sea turtles can maintain body temperatures above ambient."
            ],
            "conclusion": "Are leatherback sea turtles warm-blooded?",
            "correct_answer": "Cannot determine from these premises — the second premise is true (via gigantothermy) but doesn't change their classification as ectotherms. The premises don't define 'warm-blooded' consistently.",
            "mirage_signal": "A subtle definitional ambiguity: 'warm-blooded' (endothermy) vs. 'high body temperature'. A metacognitive model should flag this equivocation.",
            "difficulty": 5
        }
    }
]

def build_syllogism_tasks():
    tasks = []
    for i, pair in enumerate(SYLLOGISM_PAIRS):
        # Clean
        c = pair["clean"]
        premise_text = "\n".join(f"  {j+1}. {p}" for j, p in enumerate(c["premises"]))
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.SYLLOGISM_TRAP, i, "clean"),
            family=TaskFamily.SYLLOGISM_TRAP,
            variant="clean",
            prompt=f"Consider the following statements:\n{premise_text}\n\nQuestion: {c['conclusion']}\n\nFirst, state whether you are confident in your answer. Then answer.",
            correct_answer=c["correct_answer"],
            mirage_signal="N/A — this is a valid, answerable argument.",
            difficulty=c["difficulty"],
            tags=["logic", "syllogism", "deduction"]
        ))
        # Mirage
        m = pair["mirage"]
        premise_text = "\n".join(f"  {j+1}. {p}" for j, p in enumerate(m["premises"]))
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.SYLLOGISM_TRAP, i, "mirage"),
            family=TaskFamily.SYLLOGISM_TRAP,
            variant="mirage",
            prompt=f"Consider the following statements:\n{premise_text}\n\nQuestion: {m['conclusion']}\n\nFirst, state whether you are confident in your answer. Then answer.",
            correct_answer=m["correct_answer"],
            mirage_signal=m["mirage_signal"],
            difficulty=m["difficulty"],
            tags=["logic", "syllogism", "fallacy", "metacognition"]
        ))
    return tasks


# ─────────────────────────────────────────────
# TASK FAMILY 2: Unit Ghost
# ─────────────────────────────────────────────
UNIT_GHOST_PAIRS = [
    {
        "clean": "A car travels 60 miles per hour for 2 hours. How far does it travel?",
        "clean_answer": "120 miles.",
        "mirage": "A car travels 60 miles per hour for 2 hours, then switches to driving at 100 km/h for 1 more hour. How far does it travel in total?",
        "mirage_answer": "Cannot be computed without knowing the conversion or specifying a single unit. If miles: ~62.1 + 60 = ~182.1 miles. If km: ~96.6 + 100 = ~196.6 km. The mixed units require a flag.",
        "mirage_signal": "Miles and kilometers are mixed without conversion. The model should request clarification or flag the unit inconsistency before computing.",
        "difficulty": 3
    },
    {
        "clean": "A solution contains 5 grams of salt dissolved in 100 mL of water. What is the concentration in g/mL?",
        "clean_answer": "0.05 g/mL.",
        "mirage": "A solution contains 5 grams of salt dissolved in 100 mL of water. You add 50 cc of pure water. What is the new concentration?",
        "mirage_answer": "The answer is 5g / 150mL = 0.0333 g/mL — however, a metacognitive model should note that 'cc' and 'mL' are equivalent (1 cc = 1 mL), and flag whether the test-taker knows this or whether this is intentionally testing that awareness.",
        "mirage_signal": "cc vs. mL — numerically equivalent but a potential hidden trap for those who don't know they're the same. Model should note the equivalence explicitly.",
        "difficulty": 2
    },
    {
        "clean": "A recipe calls for 250g of flour. How much flour in kg?",
        "clean_answer": "0.25 kg.",
        "mirage": "A recipe calls for 2 cups of flour. The bag says '1 cup = 120g'. You have a 500g bag. You need to make 3 batches. Do you have enough? How many grams short or over are you?",
        "mirage_answer": "Need: 3 × 2 cups × 120g = 720g. Have: 500g. Short by 220g. — but the model should flag: 'cup' measurements vary by ingredient and packing; the given conversion (120g/cup for flour) is plausible but should be verified.",
        "mirage_signal": "Volumetric-to-mass conversions for flour are density-dependent and vary with packing. The model should note this caveat rather than computing blindly.",
        "difficulty": 4
    }
]

def build_unit_ghost_tasks():
    tasks = []
    for i, pair in enumerate(UNIT_GHOST_PAIRS):
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.UNIT_GHOST, i, "clean"),
            family=TaskFamily.UNIT_GHOST,
            variant="clean",
            prompt=pair["clean"] + "\n\nNote any assumptions you make, then solve.",
            correct_answer=pair["clean_answer"],
            mirage_signal="N/A",
            difficulty=1,
            tags=["math", "units", "arithmetic"]
        ))
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.UNIT_GHOST, i, "mirage"),
            family=TaskFamily.UNIT_GHOST,
            variant="mirage",
            prompt=pair["mirage"] + "\n\nNote any assumptions you make, then solve.",
            correct_answer=pair["mirage_answer"],
            mirage_signal=pair["mirage_signal"],
            difficulty=pair["difficulty"],
            tags=["math", "units", "metacognition", "ambiguity"]
        ))
    return tasks


# ─────────────────────────────────────────────
# TASK FAMILY 3: Reference Rot
# ─────────────────────────────────────────────
REFERENCE_ROT_PAIRS = [
    {
        "clean": "Alice told Bob that she would finish the report by Friday. Did Alice promise to finish the report?",
        "clean_answer": "Yes, Alice stated she would finish the report by Friday.",
        "mirage": "Alice told Carol that she would finish the report by Friday, and then Carol told Bob. Did she promise to finish the report by Friday?",
        "mirage_answer": "Ambiguous — 'she' in the final question could refer to Alice or Carol. A metacognitive model must flag this before answering.",
        "mirage_signal": "Ambiguous pronoun 'she' — could refer to Alice (the original promiser) or Carol (the conveyor). The model should request clarification.",
        "difficulty": 3
    },
    {
        "clean": "The company launched its new product on Monday. The product was a smartphone. When did the company launch its smartphone?",
        "clean_answer": "Monday.",
        "mirage": "The company launched its new product on Monday. The product sold out by Wednesday. It was later recalled. When was it back on shelves?",
        "mirage_answer": "Cannot be determined — no information about the recall duration or return date is provided. The model should not confabulate a date.",
        "mirage_signal": "The passage provides no information about when or if the product returned to shelves. A metacognitive model should state the information is missing rather than guessing.",
        "difficulty": 3
    },
    {
        "clean": "John has 3 apples. Mary has 5 apples. How many apples do John and Mary have together?",
        "clean_answer": "8 apples.",
        "mirage": "John has 3 apples. Mary has 5. She gave him some. Now he has 5. How many does she have now, and how many did she give him?",
        "mirage_answer": "He received 2 apples (5−3=2), so she now has 5−2=3 apples. BUT: the model should flag that 'she' could be Mary or another unspecified person, and 'some' implies she gave at least 1, which is consistent.",
        "mirage_signal": "The pronoun 'she' is technically unambiguous here (only one female referent), but 'some' creating the arithmetic is worth flagging as an inference, not a stated fact.",
        "difficulty": 2
    }
]

def build_reference_rot_tasks():
    tasks = []
    for i, pair in enumerate(REFERENCE_ROT_PAIRS):
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.REFERENCE_ROT, i, "clean"),
            family=TaskFamily.REFERENCE_ROT,
            variant="clean",
            prompt=pair["clean"],
            correct_answer=pair["clean_answer"],
            mirage_signal="N/A",
            difficulty=1,
            tags=["language", "reference", "reading-comprehension"]
        ))
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.REFERENCE_ROT, i, "mirage"),
            family=TaskFamily.REFERENCE_ROT,
            variant="mirage",
            prompt=pair["mirage"],
            correct_answer=pair["mirage_answer"],
            mirage_signal=pair["mirage_signal"],
            difficulty=pair["difficulty"],
            tags=["language", "reference", "ambiguity", "metacognition"]
        ))
    return tasks


# ─────────────────────────────────────────────
# TASK FAMILY 4: Sequence Lure
# ─────────────────────────────────────────────
SEQUENCE_LURE_PAIRS = [
    {
        "clean": "What is the next number in the sequence: 2, 4, 8, 16, 32, ?",
        "clean_answer": "64 — each term doubles.",
        "mirage": "What is the next number in the sequence: 2, 4, 8, 16, 31, ?",
        "mirage_answer": "There is no single well-defined next term. The sequence looks like powers of 2 but breaks at the 5th term (31 ≠ 32). One interpretation: these are the number of regions created by n points on a circle (Motzkin's formula), giving 57 next. But the model should flag the anomaly first.",
        "mirage_signal": "31 breaks the doubling pattern. Overconfident models say '62'. Metacognitive models notice 31≠32 and flag the sequence as anomalous before guessing.",
        "difficulty": 4
    },
    {
        "clean": "What comes next: 1, 1, 2, 3, 5, 8, 13, ?",
        "clean_answer": "21 — this is the Fibonacci sequence.",
        "mirage": "What comes next: 1, 1, 2, 3, 5, 8, 12, ?",
        "mirage_answer": "Ambiguous — 12 breaks the Fibonacci pattern (should be 13). Possible continuations: 19 (Fibonacci-corrected), 20 (keeping the −1 offset), or others. Model should flag the break.",
        "mirage_signal": "12 ≠ 13 breaks Fibonacci. A metacognitive model should not blindly extend 'Fibonacci' without noting the discrepancy.",
        "difficulty": 4
    },
    {
        "clean": "Continue the pattern: A, C, E, G, ?",
        "clean_answer": "I — every other letter of the alphabet.",
        "mirage": "Continue the pattern: A, C, F, J, ?",
        "mirage_answer": "O — the gaps are +2, +3, +4, +5 (triangular increments). Model should derive the rule before applying it.",
        "mirage_signal": "Not a simple skip-one pattern. The model should detect increasing gaps rather than assuming a fixed stride.",
        "difficulty": 3
    }
]

def build_sequence_lure_tasks():
    tasks = []
    for i, pair in enumerate(SEQUENCE_LURE_PAIRS):
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.SEQUENCE_LURE, i, "clean"),
            family=TaskFamily.SEQUENCE_LURE,
            variant="clean",
            prompt=f"{pair['clean']}\n\nState your confidence level (low/medium/high) and explain the pattern before giving your answer.",
            correct_answer=pair["clean_answer"],
            mirage_signal="N/A",
            difficulty=1,
            tags=["patterns", "sequences", "induction"]
        ))
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.SEQUENCE_LURE, i, "mirage"),
            family=TaskFamily.SEQUENCE_LURE,
            variant="mirage",
            prompt=f"{pair['mirage']}\n\nState your confidence level (low/medium/high) and explain the pattern before giving your answer.",
            correct_answer=pair["mirage_answer"],
            mirage_signal=pair["mirage_signal"],
            difficulty=pair["difficulty"],
            tags=["patterns", "sequences", "anomaly", "metacognition"]
        ))
    return tasks


# ─────────────────────────────────────────────
# TASK FAMILY 5: Fact Warp
# ─────────────────────────────────────────────
FACT_WARP_PAIRS = [
    {
        "clean": "What is the approximate population of the United States?",
        "clean_answer": "Approximately 335 million (as of 2024).",
        "mirage": "The US has a population of 412 million. Given that, how many people would benefit from a policy affecting 30% of the population?",
        "mirage_answer": "Based on the figure given: 0.30 × 412M = ~123.6M. BUT the model should flag that 412 million is significantly higher than the actual US population (~335M as of 2024). A metacognitive model should note this discrepancy before computing.",
        "mirage_signal": "412 million is ~23% above the actual US population. A metacognitive model should flag the false premise rather than compute silently.",
        "difficulty": 3
    },
    {
        "clean": "Light travels at approximately 300,000 km/s. How long does it take light to travel from the Earth to the Moon (~384,400 km)?",
        "clean_answer": "About 1.28 seconds.",
        "mirage": "Light travels at 300,000 km/s. The Moon is approximately 150 million km away. How long does light take to reach it?",
        "mirage_answer": "Based on the given figure: 150,000,000 / 300,000 = 500 seconds ≈ 8.3 minutes. BUT: 150 million km is the Earth-Sun distance (1 AU), not Earth-Moon (~384,400 km). The model should flag this factual error before computing.",
        "mirage_signal": "150 million km is the Earth-Sun distance. The Earth-Moon distance is ~384,400 km. A metacognitive model should flag the factual error in the premise.",
        "difficulty": 4
    },
    {
        "clean": "If a coin is flipped 10 times, what is the probability of getting exactly 5 heads?",
        "clean_answer": "C(10,5) / 2^10 = 252/1024 ≈ 24.6%.",
        "mirage": "A study found that a fair coin, when flipped by professional magicians, lands heads 58% of the time. If a magician flips a coin 10 times, what is the expected number of heads?",
        "mirage_answer": "Using the stated 58% rate: 10 × 0.58 = 5.8. BUT the model should flag that no credible study supports a 58% rate for flipped coins; research (Diaconis et al.) shows a modest same-side bias (~51%), not 58%. The premise should be questioned.",
        "mirage_signal": "58% heads for magicians is not established in credible research. The model should flag the dubious premise rather than accept it uncritically.",
        "difficulty": 5
    }
]

def build_fact_warp_tasks():
    tasks = []
    for i, pair in enumerate(FACT_WARP_PAIRS):
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.FACT_WARP, i, "clean"),
            family=TaskFamily.FACT_WARP,
            variant="clean",
            prompt=pair["clean"],
            correct_answer=pair["clean_answer"],
            mirage_signal="N/A",
            difficulty=1,
            tags=["factual", "knowledge", "arithmetic"]
        ))
        tasks.append(Task(
            task_id=Task.make_id(TaskFamily.FACT_WARP, i, "mirage"),
            family=TaskFamily.FACT_WARP,
            variant="mirage",
            prompt=pair["mirage"],
            correct_answer=pair["mirage_answer"],
            mirage_signal=pair["mirage_signal"],
            difficulty=pair["difficulty"],
            tags=["factual", "false-premise", "metacognition", "knowledge"]
        ))
    return tasks


# ─────────────────────────────────────────────
# ASSEMBLE ALL TASKS
# ─────────────────────────────────────────────
def build_all_tasks() -> list[Task]:
    tasks = []
    tasks += build_syllogism_tasks()
    tasks += build_unit_ghost_tasks()
    tasks += build_reference_rot_tasks()
    tasks += build_sequence_lure_tasks()
    tasks += build_fact_warp_tasks()
    return tasks


if __name__ == "__main__":
    tasks = build_all_tasks()
    output = [t.to_dict() for t in tasks]
    with open("/home/claude/metamirage/data/tasks.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Generated {len(tasks)} tasks ({len([t for t in tasks if t.variant == 'clean'])} clean, {len([t for t in tasks if t.variant == 'mirage'])} mirage)")
    
    by_family = {}
    for t in tasks:
        by_family.setdefault(t.family, []).append(t.variant)
    for fam, variants in by_family.items():
        print(f"  {fam}: {variants}")
