"""
Phase 1 - Step 1: Explore the IndicMSMARCO dataset (lightweight version).
"""

from datasets import load_dataset

LANGUAGE = "hi"

print(f"Loading IndicMSMARCO ({LANGUAGE})...")
dataset = load_dataset("ai4bharat/IndicMSMARCO", LANGUAGE, split="train")

print(f"\nTotal rows: {len(dataset)}")
print(f"Column names: {dataset.column_names}")

example = dataset[0]
print("\n--- First example (all fields) ---")
for key, value in example.items():
    print(f"{key}: {value}")

# How many unique queries are there vs total rows?
# (tells us if each query has multiple passage rows attached to it)
unique_queries = len(set(dataset["query_id"]))
print(f"\nUnique query_ids: {unique_queries} out of {len(dataset)} total rows")
print(f"Avg passages per query: {len(dataset) / unique_queries:.1f}")

# Check how many passages are marked as the "correct"/selected one
selected_count = sum(1 for x in dataset["is_selected"] if x)
print(f"Rows marked is_selected=True: {selected_count}")