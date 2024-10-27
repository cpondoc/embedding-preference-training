from mteb import MTEB
import mteb
from sentence_transformers import CrossEncoder, SentenceTransformer

cross_encoder = CrossEncoder("HuggingFaceTB/fineweb-edu-classifier")
dual_encoder = SentenceTransformer("all-MiniLM-L6-v2")

tasks = mteb.get_tasks(tasks=["NFCorpus"], languages=["eng"])

subset = "default" # subset name used in the NFCorpus dataset
eval_splits = ["test"]

evaluation = MTEB(tasks=tasks)
evaluation.run(
    dual_encoder,
    eval_splits=eval_splits,
    save_predictions=True,
    output_folder="results/stage1",
)
evaluation.run(
    cross_encoder,
    eval_splits=eval_splits,
    top_k=5,
    save_predictions=True,
    output_folder="results/stage2",
    previous_results=f"results/chris/NFCorpus_{subset}_predictions.json",
)