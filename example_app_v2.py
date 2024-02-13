import os
import logging
import re
import html
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv, find_dotenv
from langchain.agents import AgentExecutor
from pydantic import BaseModel
from typing import List, Optional
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
import httpx
import uvicorn

# Set up environment variables if you are using Langsmith (no direct values provided)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")
LANGCHAIN_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT")

# Load environment variables
_ = load_dotenv(find_dotenv())
# OPENAI_API_KEY should not be hardcoded or visible; ensure it's securely stored in environment variables
openai_api_key = os.environ['OPENAI_API_KEY']

# Initialize FastAPI app
app = FastAPI()

# Configure CORS for production. You may keep your domain if it's public, or consider generalizing it
origins = ["https://yourfrontenddomain.com"] 

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

# Input Validation Function 
def is_valid_input(user_input: str) -> bool:
    pattern = "^[a-zA-Z0-9\s?.,!'\"-]*$"
    return bool(re.match(pattern, user_input))

# HTML Sanitization Function
def sanitize_html(content: str) -> str:
    return html.escape(content)

# Function to check content with OpenAI Moderation API (sensitive keys are loaded from environment variables)
async def is_flagged_by_moderation(content: str) -> bool:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logging.error("OPENAI_API_KEY not set in environment variables")
        raise HTTPException(status_code=500, detail="Server configuration error")

    headers = {"Authorization": f"Bearer {openai_api_key}"}
    data = {"input": content}

    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.openai.com/v1/moderations", json=data, headers=headers)

    if response.status_code == 200:
        response_json = response.json()
        return any(result.get("flagged", False) for result in response_json.get("results", []))
    else:
        logging.error(f"Moderation API error: {response.status_code} - {response.text}")
        return False

# Set up Langchain agent (remove or generalize any specific identifiers or paths)
embeddings = OpenAIEmbeddings()
# Ensure to put in your FAISS index path
db = FAISS.load_local("your_faiss_index_path", embeddings)
retriever = db.as_retriever(search_type="mmr")
# Generalize or remove specific tool descriptions
tool = create_retriever_tool(
    retriever,
    "generic_search_tool",
    "Description of the tool goes here"
)

tools = [tool]
# Prompt template for the chatbot
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a chatbot. Assist users with their inquiries in a helpful manner."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0, streaming=True)
llm_with_tools = llm.bind_tools(tools)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class QueryInput(BaseModel):
    input: str
    chat_history: Optional[List[str]] = None
    agent_scratchpad: Optional[List[str]] = None

@app.post("/query")
async def query(input_data: QueryInput):
    if not is_valid_input(input_data.input):
        raise HTTPException(status_code=400, detail="Invalid input")

    if await is_flagged_by_moderation(input_data.input):
        raise HTTPException(status_code=403, detail="User input flagged by moderation")

    sanitized_input = sanitize_html(input_data.input)

    agent_input = {
        "input": sanitized_input,
        "chat_history": input_data.chat_history or [],
        "agent_scratchpad": input_data.agent_scratchpad or []
    }

    response = agent_executor.invoke(agent_input)

    return response

if __name__ == "__main__":
    # Use PORT environment variable, default to a generic port if not set
    port = os.getenv('PORT', '8000')
    port = int(port)
    uvicorn.run("app:app", host="0.0.0.0", port=port)
