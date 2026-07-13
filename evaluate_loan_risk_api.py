"""
Evaluates bank loan applications for default risk using the Claude API.

For each application, sends the applicant's data to Claude and asks it to
return a risk level (low/medium/high) with a short reasoning, complementing
the existing XGBoost ML model with an LLM-based reasoning layer.

Requires the ANTHROPIC_API_KEY environment variable to be set before running.
"""

import anthropic
import json

client = anthropic.Anthropic()

# Simulated data for demonstration purposes only, not real customer records.
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


for index, application in enumerate(loan_applications, start=1):
    result = evaluate_risk(application, index)
    print(result)
    print("---")
