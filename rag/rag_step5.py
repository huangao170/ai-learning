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


# ↓↓↓ 唯一的改动：从文件读取，按空行切块 ↓↓↓
def load_chunks(path):
    """读取文档，按空行切成小块"""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chunks = []
    for part in text.split("\n\n"):
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks


documents = load_chunks("rag/handbook.txt")
print("一共切成了", len(documents), "块")
for i, chunk in enumerate(documents, start=1):
    print(f"  第{i}块：{chunk[:20]}...")
print("=" * 30)
# ↑↑↑ 唯一的改动 ↑↑↑


def embed(texts):
    response = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return [item.embedding for item in response.data]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


doc_vectors = embed(documents)


def retrieve(question, k=2):
    question_vector = embed([question])[0]

    pairs = []
    for doc, doc_vector in zip(documents, doc_vectors):
        score = cosine_similarity(question_vector, doc_vector)
        pairs.append((score, doc))

    pairs = sorted(pairs, reverse=True)
    top = pairs[:k]

    for score, doc in top:
        print(f"  {score:.3f}  {doc[:20]}...")

    return [doc for score, doc in top]


def build_prompt(question, contexts):
    context_text = "\n\n".join(contexts)
    return f"请根据以下资料回答问题。如果资料里没有答案，就说不知道。\n\n资料：\n{context_text}\n\n问题：{question}"


def ask_llm(prompt):
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


for question in ["错过班车怎么办？", "我能发剧照到朋友圈吗？", "片酬少了找谁？"]:
    print("问题：", question)
    contexts = retrieve(question, k=2)
    print("回答：", ask_llm(build_prompt(question, contexts)))
    print("-" * 30)