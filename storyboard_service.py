import asyncio
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com"  # DeepSeek 直连，不走代理

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)
MODEL = "deepseek-chat"


async def ask_json(system_prompt: str, user_prompt: str) -> dict:
    """调用大模型，返回解析好的 JSON（字典）"""
    async with asyncio.timeout(60):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
    return json.loads(response.choices[0].message.content)


async def split_scenes(story: str, shot_count: int) -> list[str]:
    """第 1 步：把故事梗概拆成若干个场景"""
    system_prompt = f"""你是短剧编剧。把用户的故事梗概拆成 {shot_count} 个按时间顺序排列的场景，
每个场景用一句话描述。输出 JSON 格式：{{"scenes": ["场景1", "场景2", ...]}}"""
    data = await ask_json(system_prompt, story)
    return data["scenes"][:shot_count]


async def generate_shot(shot_id: int, scene: str, style: str) -> dict:
    """第 2 步：为单个场景生成详细分镜"""
    system_prompt = f"""你是短剧分镜师，画面风格是“{style}”。根据场景描述输出一个镜头的分镜，JSON 格式：
{{"景别": "远景/中景/近景/特写", "画面": "画面描述", "台词": "角色台词，没有则为空字符串"}}"""
    shot = await ask_json(system_prompt, scene)
    shot["镜头"] = shot_id
    shot["场景"] = scene
    return shot


async def make_storyboard(story: str, shot_count: int, style: str) -> list[dict]:
    """完整流程：先拆场景，再并发生成每个分镜"""
    scenes = await split_scenes(story, shot_count)
    shots = await asyncio.gather(
        *(generate_shot(i, scene, style) for i, scene in enumerate(scenes, start=1))
    )
    return list(shots)