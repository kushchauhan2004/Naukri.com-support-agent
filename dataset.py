# Part 1 : Dataset design and rag core
# Task 1 : Design and validate the dataset

import random



seed = 2
random_generator = random.Random(seed)

categories = [
    "Software Engineer",
    "Data Analyst",
    "Product Manager",
    "HR Executive",
    "Sales Associate",
]

statuses = [
    "Applied",
    "Screening",
    "Interview Scheduled",
    "Offered",
    "Rejected",
]

salary_ranges = {
    "Software Engineer": (400000, 1800000),
    "Data Analyst": (300000, 1000000),
    "Product Manager": (600000, 2200000),
    "HR Executive": (250000, 800000),
    "Sales Associate": (240000, 900000),
}

category_list = categories * 10
status_list = statuses * 10

random_generator.shuffle(category_list)
random_generator.shuffle(status_list)

job_applications = []

for index in range(50):
    category = category_list[index]
    minimum_salary, maximum_salary = salary_ranges[category]

    record = {
        "record_id": index + 1,
        "category": category,
        "status": status_list[index],
        "expected_salary_inr": random_generator.randrange(
            minimum_salary, maximum_salary + 1, 10000
        ),
        "days_since_created": random_generator.randint(0, 30),
        "flagged_priority_review": random_generator.random() < 0.20,
    }

    job_applications.append(record)

JOB_APPLICATIONS = job_applications

category_counts = {}
status_counts = {}

for record in JOB_APPLICATIONS:
    category = record["category"]
    status = record["status"]

    category_counts[category] = category_counts.get(category, 0) + 1
    status_counts[status] = status_counts.get(status, 0) + 1

flagged_count = sum(
    record["flagged_priority_review"] for record in JOB_APPLICATIONS
)
flagged_percentage = flagged_count / len(JOB_APPLICATIONS) * 100

print("total records:", len(JOB_APPLICATIONS))
print("count per category:", category_counts)
print("count per status:", status_counts)
print("priority review flagged:", flagged_count)
print("priority review percentage:", round(flagged_percentage, 2), "%")

assert len(JOB_APPLICATIONS) == 50
assert all(count >= 3 for count in category_counts.values())
assert all(status in status_counts for status in statuses)
assert 10 <= flagged_percentage <= 30
assert all(0 <= record["days_since_created"] <= 30 for record in JOB_APPLICATIONS)