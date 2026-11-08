# Part 2 : LangGraph Agent with Tools, Memory & Guardrails
# Task 6 :  Build the second tool with a designed escalation score.
import math

# import dataset
from dataset import JOB_APPLICATIONS


# calculate escalation score
def calculate_escalation_score(record):

    priority = 1 if record["flagged_priority_review"] else 0

    recency = record["days_since_created"] / 30

    score = (0.6 * priority) + (0.4 * recency)

    return round(score, 3)


# calculate scores for all applications
scores = [
    calculate_escalation_score(record)
    for record in JOB_APPLICATIONS
]

# calculate 80th percentile threshold
sorted_scores = sorted(scores)

position = math.ceil(0.80 * len(sorted_scores)) - 1

threshold = sorted_scores[position]


# check application status
def check_job_application_status(record_id: str) -> dict:

    for record in JOB_APPLICATIONS:

        if str(record["record_id"]) == str(record_id):

            score = calculate_escalation_score(record)

            return {
                "record_id": str(record["record_id"]),
                "status": record["status"],
                "expected_salary_inr": record["expected_salary_inr"],
                "escalation_score": score,
                "escalation_recommended": score >= threshold
            }

    return {
        "error": "Application record not found"
    }


# display escalation information
print("total applications:", len(JOB_APPLICATIONS))

print("80th percentile threshold:", threshold)

print("\nescalation formula:")
print("score = (0.6 * priority) + (0.4 * days_since_created / 30)")

# test application records
print("\napplication status results")

for record_id in ["1", "2", "3", "4", "5", "100"]:

    result = check_job_application_status(record_id)

    print("\nrecord id:", record_id)
    print(result)
