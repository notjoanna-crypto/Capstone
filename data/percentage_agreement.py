import pandas as pd
import json
from sklearn.metrics import cohen_kappa_score

# Load human labels (CSV) 
human_df = pd.read_csv("human_label_clean_rag_acc.csv")

# Ensure correct column names
human_df = human_df.rename(columns={
    human_df.columns[0]: "question_id",
    human_df.columns[1]: "human_label"
})

#  Load LLM judge labels (JSON) 
with open("data/judged_no_rag_v1.json", "r") as f:
    llm_data = json.load(f)

llm_df = pd.DataFrame([
    {
        "question_id": item["question_id"],
        "llm_label": item["correctness"]
    }
    for item in llm_data
])

#  Merge on question_id 
merged = human_df.merge(llm_df, on="question_id")

#  Percentage Agreement 
agreement_rate = (merged["human_label"] == merged["llm_label"]).mean() * 100

#  Cohen’s Kappa 
kappa = cohen_kappa_score(
    merged["human_label"],
    merged["llm_label"]
)

#  Output 
print(f"Number of samples: {len(merged)}")
print(f"Agreement Rate: {agreement_rate:.2f}%")
print(f"Cohen’s Kappa: {kappa:.3f}")


# docker compose exec app python data/percentage_agreement.py


# results clean_rag_k3.json:
# correctness
# Agreement Rate: 93.33%
# Cohen’s Kappa: 0.867


# hallucination
# Agreement Rate: 96.67%
# Cohen’s Kappa: 0.923

# source_drift
# Agreement Rate: 100.00%
# Cohen’s Kappa: nan


# results judged_conflict_noise_rag_v1.json:

#correctness
# Agreement Rate: 96.67%
# Cohen’s Kappa: 0.933

# hallucination
#Agreement Rate: 93.33%
#Cohen’s Kappa: 0.867

# source_drift
# Agreement Rate: 96.67%
# Cohen’s Kappa: 0.783


# results judged_no_rag_v1.json:
# Agreement Rate: 90.00%
# Cohen’s Kappa: 0.783