# Part 3 : Evaluation, Observability & FastAPI Deployment
# Task 11 :  Deploy via FastAPI.
import json
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langagent import app as agent_app

# create fastapi application

api = FastAPI(
    title="job application assistant",
    description="api for the langgraph rag agent",
    version="1.0"
)


# request and response models

class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    query: str
    response: str
    route: str


class DocumentRequest(BaseModel):
    title: str
    content: str


class DocumentResponse(BaseModel):
    message: str
    title: str


# json file for added documents

document_file = "api_documents.json"


# save document

def save_document(title, content):

    if os.path.exists(document_file):

        with open(document_file, "r") as file:
            documents = json.load(file)

    else:
        documents = []

    documents.append({
        "title": title,
        "content": content
    })

    with open(document_file, "w") as file:
        json.dump(documents, file, indent=4)


# endpoint 1: ask the agent

@api.post("/ask", response_model=AskResponse)
def ask_agent(request: AskRequest):

    try:

        result = agent_app.invoke({
            "query": request.query,
            "route": "",
            "response": ""
        })

        return AskResponse(
            query=request.query,
            response=str(result["response"]),
            route=result["route"]
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# endpoint 2: add a document

@api.post("/add-document", response_model=DocumentResponse)
def add_document(request: DocumentRequest):

    try:

        save_document(
            request.title,
            request.content
        )

        return DocumentResponse(
            message="Document saved successfully",
            title=request.title
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# endpoint 3: check api status

@api.get("/")
def home():

    return {
        "message": "job application assistant api is running"
    }