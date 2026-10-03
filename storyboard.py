import asyncio
import json
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com"  # DeepSeek 直连，不走代理

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

SYSTEM_PROMPT = """你是短剧分镜师。根据用户给的剧情片段，输出一个镜头的分镜，必须是 JSON 格式：
{"景别": "远景/中景/近景/特写", "画面": "画面描述", "台词": "角色台词，没有则为空字符串"}"""

scenes = [
    "雨夜，女主被赶出豪门",
    "男主的车停在女主面前",
    "三年后，女主成为公司总裁",
]


async def generate_shot(shot_id, scene):
    try:
        async with asyncio.timeout(30):
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": scene},
                ],
                response_format={"type": "json_object"},  # 强制模型输出 JSON
            )
        shot = json.loads(response.choices[0].message.content)  # 字符串 -> 字典
        shot["镜头"] = shot_id
        return shot
    except TimeoutError:
        return {"镜头": shot_id, "错误": "超时"}
    except json.JSONDecodeError:
        return {"镜头": shot_id, "错误": "模型返回的不是合法 JSON"}


async def main():
    start = time.perf_counter()
    shots = await asyncio.gather(
        *(generate_shot(i, scene) for i, scene in enumerate(scenes, start=1))
    )
    print(json.dumps(shots, ensure_ascii=False, indent=2))
    print(f"总耗时：{time.perf_counter() - start:.1f} 秒")


asyncio.run(main())
