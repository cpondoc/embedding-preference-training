import mteb
from mteb.task_selection import results_to_dataframe

tasks = mteb.get_tasks(
    task_types=["Retrieval"], languages=["eng", "fra"], domains=["Legal"]
)

model_names = [
    "GritLM/GritLM-7B",
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "intfloat/multilingual-e5-large",
]
models = [mteb.get_model_meta(name) for name in model_names]

results = mteb.load_results(models=models, tasks=tasks)

df = results_to_dataframe(results)
print(df)