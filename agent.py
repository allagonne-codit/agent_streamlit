import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from prompts import prompt
from functions import get_weather
#load_dotenv()

# HF_TOKEN = os.environ.get("HF_TOKEN")
# print(HF_TOKEN)

print(prompt)

client = InferenceClient(
    model = None,#"meta-llama/Llama-3.2-3B-Instruct",
    #token = HF_TOKEN
    )
output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": prompt},
    ],
    stream=False,
    max_tokens=200,
    stop = ["Observation:"]
)
print(output.choices[0].message.content)
new_prompt=prompt+output.choices[0].message.content+get_weather('London')
print('new prompt : ',new_prompt)

final_output = client.chat.completions.create(
    messages=[
        {"role": "user", "content": new_prompt},
    ],
    stream=False,
    max_tokens=200,
)

print(final_output.choices[0].message.content)

