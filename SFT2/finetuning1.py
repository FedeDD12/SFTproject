import os
from together import Together

client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
dataset = "train_dataset2"

response = client.files.upload(file=f"Formatted{dataset}.jsonl")
fileId = response.model_dump()["id"]

file_metadata = client.files.retrieve(fileId)
print(file_metadata)

resp = client.fine_tuning.create(
    suffix="mathinstruct6",
    model="meta-llama/Meta-Llama-3.1-8B-Reference",
    training_file=fileId,
    n_epochs=20,
    learning_rate=1e-5,
    wandb_api_key=os.environ.get("WANDB_API_KEY"),
)

print(resp)
