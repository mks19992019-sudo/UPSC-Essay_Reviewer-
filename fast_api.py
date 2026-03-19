from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend_ import run_review


app = FastAPI()



# because our ui send json we need to covert it for our llm
class mismatch(BaseModel):
    essay:str



@app.get("/home_test")
def home():
    return {"message": "it is working"}


@app.post("/review")
def review(essay: mismatch):
    return run_review(essay.essay)
