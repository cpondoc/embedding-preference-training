"""
Script to evaluate just standard embedding model or any retrieval benchmark in MTEB.
"""

from mteb import MTEB
import mteb
from sentence_transformers import SentenceTransformer

BASE_MODEL = "Snowflake/snowflake-arctic-embed-m"
TASKS = ["ArguAna", "FiQA2018", "QuoraRetrieval"]
tasks = mteb.get_tasks(tasks=TASKS, languages=["eng"])

# Iterate through each model, set up the initial encoder
dual_encoder = SentenceTransformer(BASE_MODEL)
eval_splits = ["test"]

# Run eval by first doing the results, then reranking
for task in TASKS:
    evaluation = MTEB(tasks=[task])
    evaluation.run(
        dual_encoder,
        eval_splits=eval_splits,
        save_predictions=True,
        output_folder="noisy-results-2000/" + BASE_MODEL + "/",
        classifier_normalization="top_k",
    )
