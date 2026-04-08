"""
Grader for Task 3 (Hard): Full Workflow Decision

Scoring (total 1.0):
    0.3 — classification correctness (predicted_intent == intent)
    0.3 — reply quality (keyword match ratio from expected_keywords)
    0.2 — correct escalation decision (matches should_escalate)
    0.2 — correct resolution (matches expected resolution based on escalation)
"""


def grade(prediction: dict, ground_truth: dict) -> float:
    """Grade the hard task output.

    Args:
        prediction: Dict with keys "predicted_intent", "reply", "escalate", "resolved".
        ground_truth: Dict with keys "intent", "expected_keywords", "should_escalate", "priority".

    Returns:
        Float between 0.0 and 1.0.
    """
    score = 0.0

    # --- Classification correctness (0.3 points) ---
    predicted_intent = prediction.get("predicted_intent", "")
    true_intent = ground_truth.get("intent", "")
    if predicted_intent == true_intent:
        score += 0.3

    # --- Reply quality (0.3 points based on keyword coverage) ---
    reply_text = prediction.get("reply", "").lower()
    expected_keywords = ground_truth.get("expected_keywords", [])

    if expected_keywords:
        matched = sum(1 for kw in expected_keywords if kw.lower() in reply_text)
        keyword_ratio = matched / len(expected_keywords)
        score += 0.3 * keyword_ratio

    # --- Escalation decision correctness (0.2 points) ---
    predicted_escalate = prediction.get("escalate", False)
    true_escalate = ground_truth.get("should_escalate", False)
    if predicted_escalate == true_escalate:
        score += 0.2

    # --- Resolution correctness (0.2 points) ---
    # A case is resolved if it was not escalated and is low/medium priority
    # with a non-critical intent. If escalated, it should NOT be resolved.
    predicted_resolved = prediction.get("resolved", False)

    # Expected resolution: escalated cases are not resolved,
    # low-priority general inquiries are resolved,
    # everything else depends on the agent's logic
    if true_escalate:
        expected_resolved = False
    elif ground_truth.get("priority") == "low":
        expected_resolved = True
    else:
        # For medium/high, if not escalated, the agent may resolve
        # We accept the agent's decision as long as it's not escalated
        expected_resolved = predicted_resolved  # accept agent's call

    if predicted_resolved == expected_resolved:
        score += 0.2

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, round(score, 2)))
