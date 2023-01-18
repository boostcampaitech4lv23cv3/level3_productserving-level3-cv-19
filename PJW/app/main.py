from datetime import datetime
from uuid import UUID, uuid4
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field

from app.model import get_prediction, load_model, get_config

app = FastAPI()


class Item(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    name: str


class Items(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    results: List[Item] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


class InferenceImageResult(Item):
    name: str = "inference_image_result"
    result: Optional[List]


@app.get("/")
def main_page():
    return {"message": "Hello"}


@app.post("/results")
async def predict(files: List[UploadFile] = File(...)):
    model_age = load_model("model_age")
    model_age.eval()
    model_gender = load_model("model_gender")
    model_gender.eval()
    model_mask = load_model("model_mask")
    model_mask.eval()

    results = []
    for file in files:
        image_bytes = await file.read()
        _, label_mask = get_prediction(model_mask, image_bytes)
        _, label_gender = get_prediction(model_gender, image_bytes)
        _, label_age = get_prediction(model_age, image_bytes)

        y_hat = label_mask * 6 + label_gender * 3 + label_age
        label = get_config()['classes'][y_hat.item()]
        result = InferenceImageResult(result=label)
        results.append(result)

    items = Items(results=results)
    return items
