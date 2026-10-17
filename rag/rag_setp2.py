import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com"

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

documents = [
    "拍摄时间：每天拍摄不超过 10 小时，超过 10 小时的部分按每小时 50 元计算加班费。",
    "请假规定：请假需提前 3 天向制片主任申请，病假需提供医院证明。",
    "道具管理：贵重道具使用后须当天归还道具组，丢失照价赔偿。",
    "片酬结算：演员片酬在杀青后 15 个工作日内结清，通过银行转账支付。",
    "餐饮补贴：剧组每天提供午餐和晚餐，夜戏超过 22 点额外提供夜宵。",
]


def retrieve(question):
    """检索：找出和问题最相关的一段资料"""
    best_doc = ""
    best_score = 0
    for doc in documents:
        score = len(set(question) & set(doc))
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc


def build_prompt(question, context):
    """把资料和问题拼成提示词"""
    return f"请根据以下资料回答问题。如果资料里没有答案，就说不知道。\n\n资料：{context}\n\n问题：{question}"


def ask_llm(prompt):
    """把提示词发给 DeepSeek，返回回答"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


question = "剧组有没有年终奖？"
# 对比 1：不用 RAG，直接问
print("【直接问】", ask_llm(question))
print("-" * 30)

# 对比 2：用 RAG，先检索再问
context = retrieve(question)
prompt = build_prompt(question, context)
print("【找到的资料】", context)
print("【RAG 回答】", ask_llm(prompt))