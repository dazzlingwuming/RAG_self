# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

client = OpenAI(
    api_key='sk-8ff73cc60b3b4cc295924280890d2f12',
    base_url="https://api.deepseek.com")

response = client.invoke("你好，请介绍一下自己。")
print(response.choices[0].message.content)