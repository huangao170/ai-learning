import math
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com,api.siliconflow.cn"

# 两个客户端：DeepSeek 负责回答，硅基流动负责把文字变成向量
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
    """把一组文字变成一组向量"""
    response = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度，越接近 1 越相似"""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


# 准备阶段：把所有资料变成向量（只做一次）
doc_vectors = embed(documents)
print("每个向量有", len(doc_vectors[0]), "个数字")


def retrieve(question):
    """检索：找出和问题意思最接近的一段资料"""
    question_vector = embed([question])[0]

    best_doc = ""
    best_score = -1
    for doc, doc_vector in zip(documents, doc_vectors):
        score = cosine_similarity(question_vector, doc_vector)
        print(f"  {score:.3f}  {doc[:10]}")
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc


for question in ["拍夜戏有吃的吗？", "什么时候能拿到钱？"]:
    print("问题：", question)
    print("找到：", retrieve(question))
    print("-" * 30)