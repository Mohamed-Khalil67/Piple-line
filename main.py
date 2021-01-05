from fastapi import FastAPI,Form ,Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
import re

from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from stop_words import get_stop_words

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def check_msg(text: str) -> str:
    clf = load('model/clf.joblib')
    data_input=[re.sub('[^A-Za-z\']+', ' ', text.lower())]
    predict=clf.predict(data_input)
    if predict==0:
        return "Your message contains : Hate speech."
    elif predict==1:
        return "Your message contains : Offensive language."
    else:
        return "Your message doesn't contain hate speech neither offensive language."

@app.get('/', response_class=HTMLResponse)
def homepage(request: Request):
    return templates.TemplateResponse('index.html',{"request" : request})

@app.post('/predict/',response_class=HTMLResponse)
async def predict(request : Request, text: str = Form(...)):
    sentence = check_msg(text)
    return templates.TemplateResponse('result.html',{"request":request,"text":text,"sentence":sentence} )

if __name__ == "__main__":
    uvicorn.run(app,debug=True,host="0.0.0.0", port=8000) 