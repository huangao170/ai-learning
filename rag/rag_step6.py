import os

import chromadb
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


def load_chunks(path):
    """读取文档，按空行切成小块（没变）"""
    with open(path, encoding="utf-8") as f:
        text = f.read()

    chunks = []
    for part in text.split("\n\n"):
        part = part.strip()
        if part:
            chunks.append(part)
    return chunks


def embed(texts):
    """把一组文字变成一组向量（没变）"""
    response = embed_client.embeddings.create(model="BAAI/bge-m3", input=texts)
    return [item.embedding for item in response.data]


# ↓↓↓ 新增：连接向量数据库 ↓↓↓
db = chromadb.PersistentClient(path="rag/chroma_db")
collection = db.get_or_create_collection(
    name="handbook",
    metadata={"hnsw:space": "cosine"},   # 用余弦相似度来比较
)

# 数据库是空的，才需要存入资料（只有第一次运行时会执行）
if collection.count() == 0:
    print("数据库是空的，正在存入资料……")
    documents = load_chunks("rag/handbook.txt")
    ids = []
    for i in range(len(documents)):
        ids.append(f"chunk_{i}")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embed(documents),
    )
    print("存入了", collection.count(), "块")
else:
    print("数据库里已经有", collection.count(), "块，直接使用")
# ↑↑↑ 新增 ↑↑↑


# ↓↓↓ 改动：检索交给 Chroma ↓↓↓
def retrieve(question, k=2):
    question_vector = embed([question])[0]
    results = collection.query(
        query_embeddings=[question_vector],
        n_results=k,
    )
    docs = results["documents"][0]
    distances = results["distances"][0]

    for doc, distance in zip(docs, distances):
        print(f"  相似度 {1 - distance:.3f}  {doc[:20]}...")

    return docs
# ↑↑↑ 改动 ↑↑↑


def build_prompt(question, contexts):
    """把资料和问题拼成提示词（没变）"""
    context_text = "\n\n".join(contexts)
    return f"请根据以下资料回答问题。如果资料里没有答案，就说不知道。\n\n资料：\n{context_text}\n\n问题：{question}"


def ask_llm(prompt):
    """把提示词交给 DeepSeek（没变）"""
    response = llm_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


for question in ["错过班车怎么办？", "片酬少了找谁？"]:
    print("问题：", question)
    contexts = retrieve(question, k=2)
    print("回答：", ask_llm(build_prompt(question, contexts)))
    print("-" * 30)