#!/usr/bin/env python3
"""
Customer Support Email Triage — Main Runner

Usage:
    python run.py                  # Run all three tasks and grade
    python run.py --task easy      # Run only easy task
    python run.py --task medium    # Run only medium task
    python run.py --task hard      # Run only hard task
    python run.py --verbose        # Show full replies and details
"""

import json
import os
import sys
import argparse

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from tasks import easy_task, medium_task, hard_task
from graders import easy_grader, medium_grader, hard_grader


def load_emails() -> list[dict]:
    """Load the email dataset."""
    path = os.path.join(PROJECT_ROOT, "data", "emails.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_and_grade(emails: list[dict], task_name: str, verbose: bool = False) -> list[dict]:
    """Run a task on all emails and grade the results.

    Returns a list of result dicts, one per email.
    """
    results = []

    task_map = {
        "easy": (easy_task.run, easy_grader.grade),
        "medium": (medium_task.run, medium_grader.grade),
        "hard": (hard_task.run, hard_grader.grade),
    }

    task_fn, grade_fn = task_map[task_name]

    for email in emails:
        prediction = task_fn(email)
        score = grade_fn(prediction, email)

        result = {
            "id": email["id"],
            "subject": email["subject"],
            "true_intent": email["intent"],
            "priority": email["priority"],
            "predicted_intent": prediction.get("predicted_intent", "N/A"),
            "score": score,
        }

        if task_name in ("medium", "hard"):
            result["reply"] = prediction.get("reply", "")

        if task_name == "hard":
            result["escalate"] = prediction.get("escalate", False)
            result["true_escalate"] = email.get("should_escalate", False)
            result["resolved"] = prediction.get("resolved", False)

        if verbose:
            result["_full_prediction"] = prediction

        results.append(result)

    return results


def print_table(results: list[dict], task_name: str):
    """Print a formatted results table."""
    width = 80
    print("=" * width)
    print(f"  TASK: {task_name.upper()}")
    print("=" * width)

    # Header
    print(f"{'ID':>4} | {'Subject':<35} | {'True Intent':<18} | {'Predicted':<18} | {'Score':>5}")
    print("-" * width)

    for r in results:
        subject = r["subject"][:33] + ".." if len(r["subject"]) > 35 else r["subject"]
        score_display = f"{r['score']:.2f}"
        print(
            f"{r['id']:>4} | {subject:<35} | {r['true_intent']:<18} | "
            f"{r['predicted_intent']:<18} | {score_display:>5}"
        )

    print("-" * width)

    # Summary
    scores = [r["score"] for r in results]
    avg = sum(scores) / len(scores) if scores else 0.0
    perfect = sum(1 for s in scores if s == 1.0)
    print(f"  Total emails: {len(results)}")
    print(f"  Average score: {avg:.2f}")
    print(f"  Perfect scores: {perfect}/{len(results)}")
    print("=" * width)
    print()


def print_verbose(results: list[dict], task_name: str):
    """Print detailed output including replies."""
    print_table(results, task_name)

    for r in results:
        print(f"\n--- Email #{r['id']}: {r['subject']} ---")
        print(f"  True intent:      {r['true_intent']}")
        print(f"  Predicted intent: {r['predicted_intent']}")
        print(f"  Score:            {r['score']:.2f}")

        if task_name in ("medium", "hard") and "reply" in r:
            print(f"\n  Generated Reply:\n{r['reply']}")

        if task_name == "hard":
            esc_marker = "YES" if r["escalate"] else "no"
            esc_correct = "✓" if r["escalate"] == r["true_escalate"] else "✗"
            res_marker = "YES" if r["resolved"] else "no"
            print(f"\n  Escalate: {esc_marker} ({esc_correct})")
            print(f"  Resolved: {res_marker}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Customer Support Email Triage Runner")
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Which task(s) to run (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full replies and detailed output",
    )
    args = parser.parse_args()

    emails = load_emails()
    print(f"\nLoaded {len(emails)} emails from dataset.\n")

    tasks_to_run = []
    if args.task in ("easy", "all"):
        tasks_to_run.append("easy")
    if args.task in ("medium", "all"):
        tasks_to_run.append("medium")
    if args.task in ("hard", "all"):
        tasks_to_run.append("hard")

    for task_name in tasks_to_run:
        results = run_and_grade(emails, task_name, verbose=args.verbose)
        if args.verbose:
            print_verbose(results, task_name)
        else:
            print_table(results, task_name)

    # Overall summary if running all
    if args.task == "all":
        print("=" * 80)
        print("  OVERALL SUMMARY")
        print("=" * 80)
        for task_name in ["easy", "medium", "hard"]:
            results = run_and_grade(emails, task_name)
            scores = [r["score"] for r in results]
            avg = sum(scores) / len(scores) if scores else 0.0
            perfect = sum(1 for s in scores if s == 1.0)
            print(f"  {task_name.upper():>8}: avg={avg:.2f}  |  perfect={perfect}/{len(results)}")
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
