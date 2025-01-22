import json
import csv
from together.utils import check_file

dataset="train_dataset2"
old_file_path= f"{dataset}.csv"
new_file_path= f"Formatted{dataset}.jsonl"

with open(old_file_path, "r", encoding="utf-8") as old_file:
    old_data= csv.DictReader(old_file)

llama_format = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{model_answer}<|eot_id|>
"""
formatted_data = []
system_prompt = "You're a helpful assistant that answers math questions."

with open(old_file_path, "r", encoding="utf-8") as old_file:
    old_data= csv.DictReader(old_file)

    with open(new_file_path, "w", encoding="utf-8") as new_file:
        for piece in old_data:
            temp_data = {
                "text": llama_format.format(
                    system_prompt=system_prompt,
                    user_question=piece["Question"],
                    model_answer=piece["Answer"],
                )
            }
            new_file.write(json.dumps(temp_data))
            new_file.write("\n")

report = check_file(new_file_path)
print(report)
assert report["is_check_passed"] == True
