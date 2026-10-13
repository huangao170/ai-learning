import json
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
os.environ["NO_PROXY"] = os.environ.get("NO_PROXY", "") + ",api.deepseek.com"

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)


def split_scenes(story):
    """第 1 步：把故事拆成场景（现在真的问大模型了）"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": '把故事拆成 3 个场景，输出 JSON：{"scenes": ["场景1", "场景2", "场景3"]}'},
            {"role": "user", "content": story},
        ],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content   # 大模型返回的文字
    print("大模型返回的原始文字：", text)
    data = json.loads(text)                      # 文字 -> 字典
    return data["scenes"]                        # 取出场景列表

def generate_shot(shot_id, scene):
    """第 2 步：为一个场景生成分镜（现在也问大模型了）"""
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": '根据场景写一个分镜，输出 JSON：{"景别": "远景/中景/近景/特写", "画面": "画面描述", "台词": "台词，没有就写空字符串"}'},
            {"role": "user", "content": scene},
        ],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content   # 大模型返回的文字
    shot = json.loads(text)                      # 文字 -> 字典
    shot["镜头"] = shot_id                       # 往字典里添加编号
    return shot

def make_storyboard(story):
    """导演：先拆场景，再给每个场景生成分镜（没变）"""
    scenes = split_scenes(story)

    shots = []
    for i, scene in enumerate(scenes, start=1):
        shot = generate_shot(i, scene)
        shots.append(shot)

    return shots


result = make_storyboard("被赶出豪门的女主三年后成为总裁，回来复仇")
print(result)