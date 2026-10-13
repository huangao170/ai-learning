import asyncio
import json
import os
import time

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com"

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


async def split_scenes(story):
    """第 1 步：把故事拆成场景"""
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": '把故事拆成 3 个场景，输出 JSON：{"scenes": ["场景1", "场景2", "场景3"]}'},
            {"role": "user", "content": story},
        ],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    data = json.loads(text)
    return data["scenes"]


async def generate_shot(shot_id, scene):
    """第 2 步：为一个场景生成分镜"""
    print(f"镜头{shot_id} 开始生成")
    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": '根据场景写一个分镜，输出 JSON：{"景别": "远景/中景/近景/特写", "画面": "画面描述", "台词": "台词，没有就写空字符串"}'},
            {"role": "user", "content": scene},
        ],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content
    shot = json.loads(text)
    shot["镜头"] = shot_id
    print(f"镜头{shot_id} 生成完毕")
    return shot


async def make_storyboard(story):
    """导演：先拆场景，再同时生成所有分镜"""
    scenes = await split_scenes(story)

    tasks = []
    for i, scene in enumerate(scenes, start=1):
        tasks.append(generate_shot(i, scene))

    shots = await asyncio.gather(*tasks)
    return shots


start = time.time()
result = asyncio.run(make_storyboard("被赶出豪门的女主三年后成为总裁，回来复仇"))
print(result)
print("总耗时：", round(time.time() - start, 1), "秒")