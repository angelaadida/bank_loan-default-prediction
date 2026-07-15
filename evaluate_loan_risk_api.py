"""
Evaluates bank loan applications for default risk using the Claude API.

For each application, sends the applicant's data to Claude and asks it to
return a risk level (low/medium/high) with a short reasoning, complementing
the existing XGBoost ML model with an LLM-based reasoning layer.

Results (including any failed applications) are logged to report.json
instead of only being printed, so the script can run unattended as part
of an automated workflow.

High-risk applications are flagged with a "pending_approval" status.
No notification or action is ever taken automatically on a high-risk
result — a human reviewer must check report.json and approve manually
before any follow-up communication is sent.

Requires the ANTHROPIC_API_KEY environment variable to be set before running.
"""

import anthropic
import json
from datetime import datetime

client = anthropic.Anthropic()

with open("loan_applications_sample.json", "r") as f:
    loan_applications = json.load(f)


def evaluate_risk(application, index):
    message = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=200,
        system="""You are a bank loan risk assessment system.
Input data follows the Kaggle "Bank Loan Data" dataset schema (14 standard columns).
Respond ONLY in the following JSON format, with no extra text:
{"index": ..., "risk_level": "low/medium/high", "reason": "..."}""",
        messages=[
            {"role": "user", "content": f"Loan application #{index}: {json.dumps(application)}"}
        ]
    )
    return message.content[0].text


results = []
pending_approval_count = 0  # count of applications awaiting manual review

for index, application in enumerate(loan_applications, start=1):
    try:
        raw_result = evaluate_risk(application, index)
        parsed_result = json.loads(raw_result)

        # Human Approval gate: only marks status, takes no automatic action
        if parsed_result["risk_level"] == "high":
            parsed_result["status"] = "pending_approval"
            pending_approval_count += 1
        else:
            parsed_result["status"] = "auto_logged"

        results.append(parsed_result)
        print(f"Processed application #{index}: {parsed_result['risk_level']} ({parsed_result['status']})")
    except Exception as e:
        results.append({
            "index": index,
            "risk_level": "error",
            "status": "error",
            "reason": f"Processing error: {str(e)}"
        })
        print(f"Application #{index} failed, logged as error, continuing with next application")

report = {
    "generated_at": datetime.now().isoformat(),
    "total_applications": len(loan_applications),
    "pending_approval_count": pending_approval_count,  # total count of high-risk applications for review
    "results": results
}

with open("report.json", "w") as f:
    json.dump(report, f, indent=2)

# Explicit notice: never claims an alert was sent, only that review is needed
print(f"\nDone. Report saved to report.json")
if pending_approval_count > 0:
    print(f"⚠️  {pending_approval_count} application(s) flagged as high risk and awaiting your review.")
    print("No alerts have been sent automatically. Please review report.json and approve manually.")