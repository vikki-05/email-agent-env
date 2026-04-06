"""
Grader for Task 1 (Easy): Intent Classification

Scoring:
    1.0 if predicted_intent == true_intent
    0.0 otherwise
"""


def grade(prediction: dict, ground_truth: dict) -> float:
    """Grade the easy task output.

    Args:
        prediction: Dict with key "predicted_intent".
        ground_truth: Dict with key "intent" (the true intent).

    Returns:
        1.0 if intents match exactly, 0.0 otherwise.
    """
    predicted = prediction.get("predicted_intent", "")
    true_intent = ground_truth.get("intent", "")

    if predicted == true_intent:
        return 1.0
    return 0.0
