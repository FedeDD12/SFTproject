
import json
from together import AsyncTogether
import os
import asyncio
import pandas as pd

async_together_client = AsyncTogether(api_key=os.environ.get("TOGETHER_API_KEY"))

evaluator_model = "meta-llama/Llama-3-70b-chat-hf"
eval_dataset = "EvalDataset-1000.json"

eval_dataset=pd.read_csv("eval_dataset.csv", lines=True)
eval_dataset=Dataset.from_pandas(eval_dataset)
groundTruthAnswer=dataset["Answer"]

results_noSFT=pd.read_csv("results_noSFT.csv", lines=True)
results_noSFT=Dataset.from_pandas(results_noSFT)
baseModelAnswer=results_noSFT["Generated Text"]

results=pd.read_csv("results.csv", lines=True)
results=Dataset.from_pandas(results)
finetunedModelAnswer=results_noSFT["Generated Text"]

results_instruct=pd.read_csv("results_instruct.csv", lines=True)
results_instruct=Dataset.from_pandas(results_instruct)
finetunedModelInstructAnswer=results_noSFT["Generated Text"]

baseModelCount = 0
fineTunedModelCount = 0
fineTunedModelInstructCount= 0
badResponses = 0
numErrors = 0

async def evalCompletion(groundTruthAnswer, modelAnswer):
    isAccurate = await async_together_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You will be given a ground truth answer and a model answer. Please output ACCURATE if the model answer matches the ground truth answer or INACCURATE otherwise. Please only return ACCURATE or INACCURATE. It is very important for my job that you do this.",
            },
            {
                "role": "user",
                "content": f"""{groundTruthAnswer}
                               {modelAnswer}""",
            },
        ],
        model=evaluator_model,
    )
    if isAccurate.choices[0].message.content == "ACCURATE":
        return 1, 0
    elif isAccurate.choices[0].message.content == "INACCURATE":
        return 0, 0
    else:
        return 0, 1  # Return 1 for badResponses

for result in results:
    try:
        (baseModelCount_inc, fineTunedModelCount_inc, fineTunedModelInstructCount_inc) = await asyncio.gather(
            evalCompletion(result["groundTruthAnswer"], result["baseModelAnswer"]),
            evalCompletion(result["groundTruthAnswer"], result["finetunedModelAnswer"]),
            evalCompletion(result["groundTruthAnswer"], result["finetunedModelInstructAnswer"]))

        baseModelCount += baseModelCount_inc[0]
        fineTunedModelCount += fineTunedModelCount_inc[0]
        fineTunedModelInstuctCount += fineTunedModelInstructCount_inc[0]

        badResponses += (baseModelCount_inc[1] + fineTunedModelCount_inc[1] + fineTunedModelInstructCount_inc[1])
    except Exception as e:
        numErrors += 1

print("Base model (Llama-3.1-8b): ", f"{baseModelCount / len(results) * 100}%")
print("Fine-tuned model (Llama-3.1-8b-math): ",f"{fineTunedModelCount / len(results) * 100}%")
print("Fine-tuned Instruct model (Llama-3.1-8b-Instruct-math): ", f"{fineTunedModelInstructCount / len(results) * 100}%")

