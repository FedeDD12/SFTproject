import ollama
import pandas as pd
import csv

dataset=pd.read_json(path_or_buf="wikipedia_en_mathematics_nopic_2023-08_v0.2.jsonl", lines=True)

dataset=dataset.to_dict()
qa_pairs=[]

with open('questions_and_answers.csv', 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Question', 'Answer'])  # Write the header row


for i, text in enumerate(dataset["text"].values()):
  # print(text)
  # print('Generate me a question and an answer about the content of the following article "{}" in the following format:\
      # Question:\
      # Answer:'.format(text))
  response = ollama.chat(model='llama3.1:8b-instruct-q8_0', messages=[
    {
      'role': 'user',
      'content': 'Generate me only one question and answer pair about the content of the following article "{}".\
      The response must be in the following format:\
      "**Question:** <question text>\
      **Answer:** <answer text>"\
      Do not add an additional newline between the question and the answer.\
      Do not add any additional text other than the question and answer, do not offer other information.\
      The question must not be about mathematicians\' life.\
      If the article is about a mathematicians\' life do not generate anything.\
      The question must be a mathematical fact.\
      The answer must be as complete as possible and correlated by an explaination.\
      Do not give questions and answers about books. If the article is only about a book do not write anything.\
      Do not give answers that are too simple.\
      In the answer do not add any newline.\
      Write the answer in only one line.\
      Do not generate any list. \
      Assume that the person reading the question and answer has no access to the article and has no prior knowledge of the topic. Provide the necessary context.\
      '.format(text),
    },
  ])

  print(i, response)
  # Extract the question and answer from the response
  qa_text = response['message']['content']

  try:
    question, answer = qa_text.split('**Answer:**')

    question=question.replace("**Question:**", "")
  except Exception as e:
    print("Error {} encountered...skipping...".format(e))
    continue
    
  # Append the question and answer to the list as a tuple

  qa_pairs.append((question.strip(), answer.strip()))

  # Write the question-answer pairs to a CSV file
  with open('questions_and_answers.csv', 'a', newline='', encoding='utf-8') as file:
      writer = csv.writer(file)
      writer.writerow(qa_pairs[-1])  # Write the question-answer pairs

