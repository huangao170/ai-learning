from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from storyboard_service import make_storyboard

app = FastAPI()


@app.get("/")
async def hello():
    return {"message": "你好，这是我的第一个 API"}


@app.get("/shot/{shot_id}")
async def get_shot(shot_id: int):
    return {"镜头": shot_id, "画面": "测试画面"}


class StoryRequest(BaseModel):
    """用户提交的数据格式"""
    story: str = Field(min_length=5, max_length=500)   # 故事梗概，5~500 字
    shot_count: int = Field(default=3, ge=1, le=10)    # 镜头数，默认 3，范围 1~10
    style: str = "真人"                                 # 风格，默认“真人”


@app.post("/storyboard")
async def create_storyboard(req: StoryRequest):
    try:
        shots = await make_storyboard(req.story, req.shot_count, req.style)
    except TimeoutError:
        raise HTTPException(status_code=504, detail="大模型响应超时，请稍后重试")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"生成分镜失败：{e}")
    return {"梗概": req.story, "分镜": shots}