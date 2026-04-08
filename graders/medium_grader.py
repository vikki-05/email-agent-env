"""
Grader for Task 2 (Medium): Intent Classification + Reply Generation

Scoring (total 1.0):
    0.5 — correct intent classification
    0.5 — reply quality (keyword coverage from expected_keywords)
"""


def grade(prediction: dict, ground_truth: dict) -> float:
    """Grade the medium task output.

    Args:
        prediction: Dict with keys "predicted_intent" and "reply".
        ground_truth: Dict with keys "intent" and "expected_keywords".

    Returns:
        Float between 0.0 and 1.0.
    """
    score = 0.0

    # --- Intent correctness (0.5 points) ---
    predicted = prediction.get("predicted_intent", "")
    true_intent = ground_truth.get("intent", "")
    if predicted == true_intent:
        score += 0.5

    # --- Reply quality (0.5 points based on keyword coverage) ---
    reply_text = prediction.get("reply", "").lower()
    expected_keywords = ground_truth.get("expected_keywords", [])

    if expected_keywords:
        matched = sum(1 for kw in expected_keywords if kw.lower() in reply_text)
        keyword_ratio = matched / len(expected_keywords)
        score += 0.5 * keyword_ratio

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, round(score, 2)))
