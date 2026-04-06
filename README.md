# Customer Support Email Triage System

An OpenEnv environment for evaluating AI agents on customer support email triage tasks. This system simulates a real-world support inbox where an agent must classify, respond to, and manage incoming emails.

---

## Project Structure

```
openenv-customer-support/
├── data/
│   └── emails.json              # 12 realistic customer support emails
├── tasks/
│   ├── easy_task.py             # Task 1: Intent classification only
│   ├── medium_task.py           # Task 2: Classification + reply generation
│   └── hard_task.py             # Task 3: Full workflow (classify + reply + escalate + close)
├── graders/
│   ├── easy_grader.py           # Exact intent match → 0.0 or 1.0
│   ├── medium_grader.py         # 0.5 intent + 0.5 keyword match
│   └── hard_grader.py           # 0.3 + 0.3 + 0.2 + 0.2 breakdown
├── models/
│   └── agent.py                 # Rule-based classifier + reply generator
├── run.py                       # CLI runner
└── README.md
```

---

## Tasks

### Task 1 — Easy: Intent Classification
- **Input:** Email text
- **Output:** Predicted intent (`refund_request`, `delivery_issue`, `complaint`, `technical_issue`, `general_inquiry`)
- **Grader:** 1.0 if correct, 0.0 if wrong

### Task 2 — Medium: Classification + Reply
- **Input:** Email text
- **Output:** Predicted intent + generated reply text
- **Grader:** 0.5 for correct intent + 0.5 for keyword coverage in reply

### Task 3 — Hard: Full Workflow
- **Input:** Email text + priority
- **Output:** Intent + reply + escalation decision + resolution status
- **Grader:** 0.3 classification + 0.3 reply keywords + 0.2 escalation + 0.2 resolution

---

## Quick Start

No dependencies required — pure Python 3.10+.

```bash
# Run all tasks
python run.py

# Run a specific task
python run.py --task easy
python run.py --task medium
python run.py --task hard

# Verbose mode (shows full replies)
python run.py --verbose
```

---

## Dataset

12 realistic customer support emails across 5 intents and 3 priority levels:

| Intent              | Count | Priority Mix         |
|---------------------|-------|----------------------|
| `refund_request`    | 3     | medium, high, medium |
| `delivery_issue`    | 2     | high, high           |
| `complaint`         | 2     | high, medium         |
| `technical_issue`   | 3     | medium, medium, high |
| `general_inquiry`   | 2     | low, low             |

---

## Design Decisions

- **Zero external dependencies** — uses only Python standard library
- **Fully deterministic** — no randomness; identical results every run
- **All scores in [0.0, 1.0]** — clamped and rounded to 2 decimal places
- **Rule-based agent** — keyword matching for classification, template-based replies
- **Pluggable architecture** — swap in your own agent model by modifying `models/agent.py`

---

## Extending

To use your own AI model instead of the rule-based agent:

1. Modify `models/agent.py` to call your model's API
2. Keep the same method signatures (`classify`, `reply`, `decide_escalation`, etc.)
3. Re-run `python run.py` — graders work with any agent that follows the interface
