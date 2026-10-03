import asyncio
import random
import time

async def call_llm(shot_id):
    """模拟调用大模型：随机耗时 1~3 秒"""
    await asyncio.sleep(random.uniform(1, 3))
    if shot_id == 3:
        raise ValueError("模型返回格式错误")
    return f"镜头{shot_id}：分镜描述..."

async def generate_shot(shot_id):
    try:
        async with asyncio.timeout(2.5):          # 单个请求最多等 2.5 秒
            return await call_llm(shot_id)
    except TimeoutError:
        return f"镜头{shot_id}：超时"

async def main():
    start = time.perf_counter()
    results = await asyncio.gather(
        *(generate_shot(i) for i in range(1, 6)),
        return_exceptions=True                    # 一个出错不影响其他
    )
    for r in results:
        print(r)
    print(f"总耗时：{time.perf_counter() - start:.1f} 秒")   # 约 2~3 秒，而不是 5~15 秒

asyncio.run(main())