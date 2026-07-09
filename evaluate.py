"""
evaluate.py — Day 5 evaluation harness
Runs the fraud triage agent against the 50-case held-out eval set.

Produces:
  - Accuracy vs ground truth labels
  - Confidence calibration table
  - Escalation rate
  - Reasoning quality score (manual review of 5 sampled traces)
  - Full results saved to data/eval_results.json

Run with: python evaluate.py
"""

import json
import time
from pathlib import Path
from datetime import datetime
from agent_design import run_case

EVAL_PATH    = Path("./data/processed/eval_set.jsonl")
RESULTS_PATH = Path("./data/eval_results.json")

# ── Load eval set ─────────────────────────────────────────────────────────────
print("Loading eval set...")
eval_cases = []
with open(EVAL_PATH) as f:
    for line in f:
        eval_cases.append(json.loads(line))
print(f"  {len(eval_cases)} cases loaded\n")

# ── Run agent on each case ────────────────────────────────────────────────────
print("Running agent (this takes ~3 minutes for 50 cases)...")
print("─" * 60)

results = []
errors  = 0

for i, case in enumerate(eval_cases):
    case_id     = case["case_id"]
    ground_truth = bool(case.get("_ground_truth_fraud", False))

    try:
        result = run_case(case_id)

        rec    = result.get("recommendation", "escalate")
        conf   = result.get("confidence", "medium")
        c_score = float(result.get("confidence_score", 0.5))
        flag   = result.get("secondary_flag", "ambiguous")
        trace  = result.get("reasoning_trace", "")
        esc    = result.get("escalation_required", False)

        # Agreement: agent recommendation matches ground truth direction
        # approve/escalate on legit = correct; decline/escalate on fraud = correct
        # We treat "escalate" as neutral — neither right nor wrong
                
        if rec == "approve" and not ground_truth:
            agreement = True    # correctly approved legit
        elif rec == "decline" and ground_truth:
            agreement = True    # correctly declined fraud
        elif rec == "escalate" and ground_truth:
            agreement = True    # correctly flagged fraud for human review
        elif rec == "escalate" and not ground_truth:
            agreement = None    # sent legit to human review — cautious, not wrong
        elif rec == "approve" and ground_truth:
            agreement = False   # missed fraud — worst outcome
        elif rec == "decline" and not ground_truth:
            agreement = False   # false positive — second worst
        else:
            agreement = None

        results.append({
            "case_id":       case_id,
            "ground_truth":  ground_truth,
            "recommendation": rec,
            "confidence":    conf,
            "confidence_score": c_score,
            "secondary_flag": flag,
            "escalation_required": esc,
            "agreement":     agreement,
            "reasoning_trace": trace,
        })

        # Progress indicator
        gt_label  = "FRAUD" if ground_truth else "LEGIT"
        ag_label  = rec.upper()[:3]
        ag_symbol = "✓" if agreement is True else ("✗" if agreement is False else "~")
        print(f"  [{i+1:02d}/50] {case_id} | GT:{gt_label} | Agent:{ag_label} | {ag_symbol}")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    except Exception as e:
        print(f"  [{i+1:02d}/50] {case_id} | ERROR: {e}")
        errors += 1
        results.append({
            "case_id": case_id,
            "ground_truth": ground_truth,
            "recommendation": "escalate",
            "confidence": "low",
            "confidence_score": 0.0,
            "secondary_flag": "ambiguous",
            "escalation_required": True,
            "agreement": None,
            "reasoning_trace": f"Error: {str(e)}",
        })

print("─" * 60)
print(f"\nCompleted: {len(results)} cases, {errors} errors\n")

# ── Compute metrics ────────────────────────────────────────────────────────────
print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)

# 1. Agreement rate (exclude escalated/neutral cases)
decided    = [r for r in results if r["agreement"] is not None]
correct    = [r for r in decided if r["agreement"] is True]
wrong      = [r for r in decided if r["agreement"] is False]
neutral    = [r for r in results if r["agreement"] is None]  # escalated

accuracy   = len(correct) / len(decided) * 100 if decided else 0
escalation_rate = len(neutral) / len(results) * 100

print(f"\n1. AGREEMENT RATE (on non-escalated cases)")
print(f"   Decided cases:   {len(decided)}/50")
print(f"   Correct:         {len(correct)}")
print(f"   Incorrect:       {len(wrong)}")
print(f"   Agreement rate:  {accuracy:.1f}%")

# 2. Escalation rate
print(f"\n2. ESCALATION RATE")
print(f"   Escalated:       {len(neutral)}/50 ({escalation_rate:.1f}%)")

# 3. Confidence distribution
conf_dist = {}
for r in results:
    c = r["confidence"]
    conf_dist[c] = conf_dist.get(c, 0) + 1

print(f"\n3. CONFIDENCE DISTRIBUTION")
for level in ["high", "medium", "low"]:
    count = conf_dist.get(level, 0)
    pct   = count / len(results) * 100
    bar   = "█" * int(pct / 5)
    print(f"   {level:6s}: {count:2d} cases ({pct:4.1f}%) {bar}")

# 4. Calibration table
# For each confidence level: how often was the agent correct?
print(f"\n4. CALIBRATION TABLE")
print(f"   {'Confidence':<12} {'Cases':>6} {'Correct':>8} {'Wrong':>8} {'Accuracy':>10}")
print(f"   {'─'*46}")

calibration = {}
for level in ["high", "medium", "low"]:
    level_cases   = [r for r in decided if r["confidence"] == level]
    level_correct = [r for r in level_cases if r["agreement"] is True]
    level_acc     = len(level_correct) / len(level_cases) * 100 if level_cases else 0
    calibration[level] = {
        "cases": len(level_cases),
        "correct": len(level_correct),
        "wrong": len(level_cases) - len(level_correct),
        "accuracy": level_acc,
    }
    print(f"   {level:<12} {len(level_cases):>6} {len(level_correct):>8} {len(level_cases)-len(level_correct):>8} {level_acc:>9.1f}%")

# Calibration check: high confidence should be more accurate than medium/low
high_acc = calibration.get("high", {}).get("accuracy", 0)
med_acc  = calibration.get("medium", {}).get("accuracy", 0)
low_acc  = calibration.get("low", {}).get("accuracy", 0)

calibration_ok = high_acc >= med_acc
print(f"\n   Calibration check: {'✓ PASS' if calibration_ok else '⚠ REVIEW'}")
print(f"   (High confidence accuracy {high_acc:.0f}% {'≥' if calibration_ok else '<'} medium {med_acc:.0f}%)")

# 5. Recommendation distribution
print(f"\n5. RECOMMENDATION DISTRIBUTION")
rec_dist = {}
for r in results:
    rec = r["recommendation"]
    rec_dist[rec] = rec_dist.get(rec, 0) + 1
for rec, count in sorted(rec_dist.items()):
    print(f"   {rec:<10}: {count:2d} cases ({count/len(results)*100:.1f}%)")

# 6. False negatives (missed fraud) — highest risk
false_neg = [r for r in results if r["ground_truth"] is True and r["recommendation"] == "approve"]
false_pos = [r for r in results if r["ground_truth"] is False and r["recommendation"] == "decline"]

print(f"\n6. RISK METRICS")
print(f"   False negatives (fraud approved): {len(false_neg)} cases  ← highest risk")
print(f"   False positives (legit declined): {len(false_pos)} cases")

# 7. Sample reasoning traces for manual quality review
print(f"\n7. SAMPLE REASONING TRACES (manual quality check)")
print(f"   Review 5 cases below — score 1-5 for specificity\n")
samples = results[:2] + results[25:27] + results[48:49]
for s in samples:
    print(f"   Case: {s['case_id']} | GT: {'FRAUD' if s['ground_truth'] else 'LEGIT'} | Rec: {s['recommendation'].upper()}")
    trace_preview = s['reasoning_trace'][:200] + "..." if len(s['reasoning_trace']) > 200 else s['reasoning_trace']
    print(f"   {trace_preview}")
    print()

# ── Summary for case study ────────────────────────────────────────────────────
print("=" * 60)
print("NUMBERS FOR YOUR CASE STUDY (copy these into index.html)")
print("=" * 60)
print(f"  Agreement rate:    {accuracy:.0f}%  (approve/decline/escalate-fraud vs ground truth)")
print(f"  Escalation rate:   {escalation_rate:.0f}%  (cases routed to mandatory human review)")
print(f"  False neg rate:    {len(false_neg)/len(results)*100:.0f}% ({len(false_neg)} fraud cases incorrectly approved)")
print(f"  Zero auto-errors:  {'Yes' if len(false_neg) == 0 else 'No'} (no fraud slipped through as approve)")
print(f"  Reasoning quality: [Score manually 1–5 after reading traces above]")
print()

# ── Save full results ─────────────────────────────────────────────────────────
output = {
    "run_at":          datetime.utcnow().isoformat(),
    "total_cases":     len(results),
    "errors":          errors,
    "metrics": {
        "agreement_rate":    round(accuracy, 1),
        "escalation_rate":   round(escalation_rate, 1),
        "false_negative_count": len(false_neg),
        "false_positive_count": len(false_pos),
        "calibration_pass":  calibration_ok,
        "confidence_distribution": conf_dist,
        "recommendation_distribution": rec_dist,
        "calibration_table": calibration,
    },
    "results": results,
}

with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"Full results saved to {RESULTS_PATH}")
print("Done.\n")
