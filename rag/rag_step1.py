# 知识库：一家虚构的短剧公司的内部规定（大模型不可能知道这些内容）
documents = [
    "拍摄时间：每天拍摄不超过 10 小时，超过 10 小时的部分按每小时 50 元计算加班费。",
    "请假规定：请假需提前 3 天向制片主任申请，病假需提供医院证明。",
    "道具管理：贵重道具使用后须当天归还道具组，丢失照价赔偿。",
    "片酬结算：演员片酬在杀青后 15 个工作日内结清，通过银行转账支付。",
    "餐饮补贴：剧组每天提供午餐和晚餐，夜戏超过 22 点额外提供夜宵。",
]


def retrieve(question):
    """检索：找出和问题最相关的一段资料（最笨的办法：数相同的字有几个）"""
    best_doc = ""
    best_score = 0
    for doc in documents:
        score = len(set(question) & set(doc))   # 问题和资料里共同出现的字的个数
        if score > best_score:
            best_score = score
            best_doc = doc
    return best_doc


def build_prompt(question, context):
    """把资料和问题拼成提示词"""
    return f"请根据以下资料回答问题。如果资料里没有答案，就说不知道。\n\n资料：{context}\n\n问题：{question}"


question = "拍戏超时有加班费吗？"
context = retrieve(question)
prompt = build_prompt(question, context)

print("找到的资料：", context)
print("-" * 30)
print("拼好的提示词：")
print(prompt)