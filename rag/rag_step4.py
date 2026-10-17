import math
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com,api.siliconflow.cn"

llm_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
embed_client = OpenAI(
    api_key=os.getenv("SILICONFLOW_API_KEY"),
    base_url="https://api.siliconflow.cn/v1",
)

documents = [
    "拍摄时间：每天拍摄不超过 10 小时，超过 10 小时的部分按每小时 50 元计算加班费。",
    "请假规定：请假需提前 3 天向制片主任申请，病假需提供医院证明。",
    "道具管理：贵重道具使用后须当天归还道具组，丢失照价赔偿。",
    "片酬结算：演员片酬在杀青后 15 个工作日内结清，通过银行转账支付。",
    "餐饮补贴：剧组每天提供午餐和晚餐，夜戏超过 22 点额外提供夜宵。",
]


def embed(texts):
    """把一组文字变成一组向量（没变）"""
    response = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度（没变）"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


doc_vectors = embed(documents)


# ① 改动：不再只取最高的 1 段，而是取分数最高的前 k 段
def retrieve(question, k=2):
    question_vector = embed([question])[0]

    pairs = []
    for doc, doc_vector in zip(documents, doc_vectors):
        score = cosine_similarity(question_vector, doc_vector)
        pairs.append((score, doc))

    pairs = sorted(pairs, reverse=True)   # 按分数从高到低排序
    top = pairs[:k]                       # 取前 k 个

    for score, doc in top:
        print(f"  {score:.3f}  {doc}")

    return [doc for score, doc in top]    # 只返回资料文字，不要分数


# ② 改动：把多段资料拼在一起，每段占一行
def build_prompt(question, contexts):
    context_text = "\n".join(contexts)
    return f"请根据以下资料回答问题。如果资料里没有答案，就说不知道。\n\n资料：\n{context_text}\n\n问题：{question}"


# ③ 加回：把提示词交给 DeepSeek
def ask_llm(prompt):
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


question = "拍夜戏有什么福利？"
print("问题：", question)
print("检索到的资料：")
contexts = retrieve(question, k=2)
prompt = build_prompt(question, contexts)
print(prompt)
print("回答：", ask_llm(prompt))