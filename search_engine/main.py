import os
import secrets

from fastapi import FastAPI, Request, status, HTTPException, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated
from pymongo.mongo_client import MongoClient

from gui_repository import GUIRepository
from rawi_repository import RaWiRepository

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="/"), name="images")
security = HTTPBasic()
templates = Jinja2Templates(directory="templates")

processor_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-all")

gp_index_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/gui_repository/gui_repository_mix_11460.index")
gp_model_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-all/checkpoint-11460")

no_gp_index_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/gui_repository/s2w_cla_gui_repository_s2w_cla_6170.index")
no_gp_model_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-s2w-cla/checkpoint-6170")

gp_db = GUIRepository(index_path=gp_index_path, model_path=gp_model_path, processor_path=processor_path)
no_gp_db = GUIRepository(index_path=no_gp_index_path, model_path=no_gp_model_path, processor_path=processor_path)
rawi_db = RaWiRepository()

mongo_client = MongoClient(os.environ['MONGODB_URI'])

def auth(credentials):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"imt-ales-2024"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"jialiangwei"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True

@app.get("/", response_class=HTMLResponse)
def index(credentials: Annotated[HTTPBasicCredentials, Depends(security)], 
          request: Request, query: str='', dataset: str='', user: str=''):
    print(request.client.host)
    auth(credentials)
    if query:
        if dataset == 'no_gp':
            result = no_gp_db.query(query, count=10)
            images = list(map(lambda x: "/images" + x, result))
        elif dataset == "gp":
            result = gp_db.query(query, count=10)
            images = list(map(lambda x: "/images" + x, result))
        elif dataset == "rawi":
            result = rawi_db.query(query, count=10)
            images = result
    else:
        images = []

    return templates.TemplateResponse("index.html", 
                                      {"request": request, "images": images, "query": query, "dataset": dataset, "user": user})
