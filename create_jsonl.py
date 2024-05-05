"""Tạo tệp JSONL từ tệp JSON của MetaMathQA-40K.
"""

import json
import pandas as pd
import jsonlines

with open('./Data/MetaMathQA-40K.json', 'r') as input_file, open('./Data/data.jsonl', 'w', encoding='utf-8') as output_file:
    data = json.load(input_file)

    # Tạo một danh sách các đối tượng JSONL
    jsonl_data = []

    prompt = "You are a professional math solving assistant. Given a math problem, solve it step-by-step and provide a clear and concise explanation of the solution."

    for item in data:
        train_object = {
            "messages": [
                {
                    "role": "system",
                    "content": prompt

                },
                {
                    "role": "user",
                    "content": item['query']

                },
                {
                    "role": "model",
                    "content": item['response']
                }
            ]
        }
        jsonl_data.append(train_object)

    writer = jsonlines.Writer(output_file)
    writer.write_all(jsonl_data)
