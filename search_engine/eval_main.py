import os
import secrets
import random

from fastapi import FastAPI, Request, status, HTTPException, Form, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from urllib.parse import urlencode
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

mix_index_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/gui_repository/gui_repository_mix_11460.index")
mix_model_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-all/checkpoint-11460")

rico_redraw_index_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/gui_repository/s2w_cla_gui_repository_s2w_cla_6170.index")
rico_redraw_model_path = os.path.join(os.environ['GUI_SEARCH_ROOT_PATH'], "retrieval/models/clip-base-mix-s2w-cla/checkpoint-6170")

mix_db = GUIRepository(index_path=mix_index_path, model_path=mix_model_path, processor_path=processor_path)
rico_redraw__db = GUIRepository(index_path=rico_redraw_index_path, model_path=rico_redraw_model_path, processor_path=processor_path)
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
          request: Request, query: str='', user: str=''):
    auth(credentials)
    
    if query:
        result = {
            "mix": mix_db.query(query, count=10),
            "rico_redraw": rico_redraw__db.query(query, count=10),
            "rawi": rawi_db.query(query, count=10)
        }
        images = [{"source": "mix", "rank": i, "path": "/images" + x} for i, x in enumerate(result["mix"])]
        images.extend([{"source": "rico_redraw", "rank": i, "path": "/images" + x} for i, x in enumerate(result["rico_redraw"])])
        images.extend([{"source": "rawi", "rank": i, "path": "/images" + x} for i, x in enumerate(result["rawi"])])
        random.Random(916).shuffle(images)
    else:
        images = []

    return templates.TemplateResponse("eval_index.html", 
                                      {"request": request, "images": images, "query": query, "user": user})

@app.post("/eval/", response_class=RedirectResponse)
def submit_eval(credentials: Annotated[HTTPBasicCredentials, Depends(security)], 
                request: Request, query: Annotated[str, Form()],  user: Annotated[str, Form()],
                mix_selected_image_ids: Annotated[str, Form()], mix_selected_image_paths: Annotated[str, Form()],
                rico_redraw_selected_image_ids: Annotated[str, Form()], rico_redraw_selected_image_paths: Annotated[str, Form()],
                rawi_selected_image_ids: Annotated[str, Form()], rawi_selected_image_paths: Annotated[str, Form()]):

    auth(credentials)

    record = {"query": query, "user": user, 
              "mix_selected_image_ids": mix_selected_image_ids.split(","),
              "mix_selected_image_paths": mix_selected_image_paths.split(","),
              "rico_redraw_selected_image_ids": rico_redraw_selected_image_ids.split(","),
              "rico_redraw_selected_image_paths": rico_redraw_selected_image_paths.split(","),
              "rawi_selected_image_ids": rawi_selected_image_ids.split(","),
              "rawi_selected_image_paths": rawi_selected_image_paths.split(",")}
    mongo_client['evaluation']["search_engine"].insert_one(record)
    
    redirect_url = request.url_for('index')
    query_string = urlencode({"query": "", "user": user})
    return RedirectResponse(f"{redirect_url}?{query_string}", status_code=status.HTTP_303_SEE_OTHER)    
