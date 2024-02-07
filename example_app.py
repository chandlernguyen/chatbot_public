from flask import Flask, request, jsonify, session, abort
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging
import os
import openai
from langchain import hub
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import re
import html

def convert_dicts_to_agent_format(chat_history_dicts):
    """
    Converts a list of dictionaries representing chat history into a list of agent format messages.

    Args:
        chat_history_dicts (list of dicts): The chat history in dictionary format.

    Returns:
        list: The chat history in agent message format.
    """
    chat_history = []
    for message in chat_history_dicts:
        if message["type"] == "user":
            chat_history.append(HumanMessage(content=message["content"]))
        elif message["type"] == "bot":
            chat_history.append(AIMessage(content=message["content"]))
    return chat_history

def convert_agent_format_to_dicts(chat_history):
    """
    Converts a list of agent format messages into a list of dictionaries representing chat history.

    Args:
        chat_history (list): The chat history in agent message format.

    Returns:
        list of dicts: The chat history in dictionary format.
    """
    chat_history_dicts = []
    for message in chat_history:
        message_type = "user" if isinstance(message, HumanMessage) else "bot"
        chat_history_dicts.append({"type": message_type, "content": message.content})
    return chat_history_dicts

# Define a function to validate user input
def is_valid_input(user_input):
    """
    Validates the user input against a regex pattern.

    Args:
        user_input (str): The input string to validate.

    Returns:
        bool: True if the input is valid, False otherwise.
    """
    pattern = "^[a-zA-Z0-9\s?.,!'\"-]*$"
    return re.match(pattern, user_input)

# Define a function to sanitize HTML content
def sanitize_html(content):
    """
    Escapes HTML characters in the content to prevent XSS attacks.

    Args:
        content (str): The content to sanitize.

    Returns:
        str: The sanitized content.
    """
    return html.escape(content)

# Define a function to check content against OpenAI's Moderation API
def is_flagged_by_moderation(content):
    """
    Checks the content against OpenAI's Moderation API to identify potentially unsafe content.

    Args:
        content (str): The content to check.

    Returns:
        bool: True if the content is flagged, False otherwise.
    """
    try:
        response = openai.Moderation.create(input=content)
        return response["results"][0]["flagged"]
    except Exception as e:
        logging.error(f"Moderation API error: {e}")
        return False  # or True, depending on how you want to handle API errors

app = Flask(__name__)
# Enable CORS for the '/query' endpoint, replace "query" with the actual endpoint you want to use
CORS(app, resources={r"/query": {"origins": "[YOUR_DOMAIN_HERE]"}})  # Replace [YOUR_DOMAIN_HERE] with your actual domain

app.debug = False
limiter = Limiter(app=app, key_func=get_remote_address)
# Set up logging if you want to log errors and debug information
# logging.basicConfig(filename='application.log', level=logging.DEBUG)

_ = load_dotenv(find_dotenv()) 

# Set up OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings()
db = FAISS.load_local("path/to/your/faiss_index_file", embeddings)  # Replace the path with your actual FAISS index file path
retriever = db.as_retriever(search_type="mmr")
tool = create_retriever_tool(
    retriever,
    "search_your_blog",  # Replace "search_your_blog" with a descriptive name for your tool
    "Your tool description here"  # Provide a brief description of what your tool does
)

tools = [tool]
prompt_template = ChatPromptTemplate.from_messages([
    # Customize the prompt template according to your chatbot's persona and requirements
])

llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", temperature=0) #You can replace the model name with any other model you want to use
llm_with_tools = llm.bind_tools(tools)
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt_template
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

@app.route('/query', methods=['POST']) # Replace '/query' with the actual endpoint you want to use
@limiter.limit("30 per minute")
def query(): # Replace 'query' with the actual function name you want to use
    """
    Handles POST requests to query the blog. Validates and processes user input,
    retrieves responses, and returns sanitized responses to the user.
    """
    user_input = request.json.get('user_input')

    if not user_input or not is_valid_input(user_input):
        abort(400, 'Invalid input')

    if is_flagged_by_moderation(user_input):
        return jsonify({"error": "User input flagged by moderation"}), 403

    chat_history_dicts = session.get('chat_history', [])

    try:
        chat_history = convert_dicts_to_agent_format(chat_history_dicts)
        chat_history.append(HumanMessage(content=user_input))
        result = agent_executor.invoke({"input": user_input, "chat_history": chat_history})
        chat_history.append(AIMessage(content=result["output"]))

        if is_flagged_by_moderation(result["output"]):
            return jsonify({"error": "Response flagged by moderation"}), 403

        chat_history_dicts = convert_agent_format_to_dicts(chat_history)
        session['chat_history'] = chat_history_dicts
        sanitized_response = sanitize_html(result["output"])

        logging.info("Sending back response: %s", sanitized_response)
        return jsonify({"response": sanitized_response})

    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return jsonify({"error": "Error processing your request"}), 500

@app.route('/')
def home():
    """
    Simple route to confirm the server is running.
    """
    return 'Hello, World!'

if __name__ == '__main__':
    # app.run(debug=True)