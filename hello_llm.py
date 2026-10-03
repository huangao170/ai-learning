import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 读取 .env 文件，把里面的内容放进环境变量
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com"  # DeepSeek 直连，不走代理

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个短剧编剧"},
        {"role": "user", "content": "用一句话写一个霸总短剧的开场"},
    ],
)

print(response.choices[0].message.content)