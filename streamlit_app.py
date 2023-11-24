#import time
#import logging
#import uuid
import streamlit as st
#import datetime
#from agent import get_qa_agent
#from send_feedback import send_to_pubsub, _submit_feedback
#from google.cloud import aiplatform
#from logo import add_logo
#from config import (
#    PROJECT_ID,
#    REGION,
#    BUCKET_NAME
#)
from streamlit_feedback import streamlit_feedback
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.agent_toolkits import create_conversational_retrieval_agent
from langchain.chat_models import ChatOpenAI
import os
from PIL import Image
# Load OAI Key
os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']


#from source_retriever import list_top_k_sources, get_top_k_urls
#logging.basicConfig(level=logging.INFO)

#uuid = uuid.uuid4()

#aiplatform.init(
#    project=PROJECT_ID,
#    location=REGION,
#    staging_bucket=f"gs://{BUCKET_NAME}"
#)

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

def get_qa_agent():
    # Loading data
    loader = DirectoryLoader("FAQ/", glob="*.txt")
    documents = loader.load()

#    for doc in documents: 
#        doc.metadata["name"] = doc.metadata["source"].split("\\")[1].replace("_", " ")[:-4].capitalize()
    
    # Chunking and VectorStore 
    text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # Retriever tool
    retriever = db.as_retriever()
    tool = create_retriever_tool(
        retriever,
        "search_transavia_FAQ",
        "Searches the Transavia company FAQ Documents to answer the user question regarding the company policies"
    )
    tools = [tool]

    # Agent constructor
    llm = ChatOpenAI(temperature=0)
    #print(f"Model : {llm.model_name}")
    
    agent_executor = create_conversational_retrieval_agent(
        llm,
        tools,
        verbose=False
    )
    return agent_executor

@st.cache_resource
def cached_qa_agent():
    qa_agent = get_qa_agent()
    return qa_agent

# Logo
transavia_logo = add_logo(logo_path="./logo/logo_octo.png", width=1280, height=250)
st.image(transavia_logo)


st.title("Transavia's Conversational FAQ")

qa_agent = cached_qa_agent()



st.write(f"Based on OpenAI {qa_agent.agent.llm.model_name}")

#with st.sidebar:
#    st.image(octo_logo)
#    st.caption('Select the documents you want to search in:')
#    luggage_docs = st.checkbox('luggage docs', value=True)
#    if luggage_docs:
#        pass

# Use if additional FAQ docs (cf Zeno)

#    ai_docs = st.checkbox('AI docs', value=True)
#    if ai_docs:
#        pass

#    st.caption('Search in a specific time range:')
#    docs_start_date = st.date_input("Start date", datetime.date(2020, 1, 1))
#    docs_end_date = st.date_input("End date", datetime.date.today())


if "feedback_key" not in st.session_state:
    st.session_state.feedback_key = 0

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello, I'm the Transavia FAQ Assistant, how can I help you today?"}]

if "runtime" not in st.session_state:
    st.session_state.runtime = 0

if "sources" not in st.session_state:
    st.session_state.sources = ""

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input("How can I help ?"):
    # Add prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

#    # Get answer
#    time_st = time.time()
    result = qa_agent({"input": prompt})
#    sources_str = list_top_k_sources(result)
#    st.session_state.sources = get_top_k_urls(result)
#    answer = result["result"]
    answer = result["output"]
#    documents = result["source_documents"]
#    time_end = time.time()

    # Add answer and sources
#    st.chat_message("assistant").write(answer + f" ({time_end - time_st:.2f}s)")
    st.chat_message("assistant").write(answer)
#    st.markdown(sources_str)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
#    st.session_state.runtime = round(time_end - time_st, 3)
    st.session_state.feedback_key += 1

# Feedback
feedback = streamlit_feedback(
    feedback_type="faces",
#    on_submit=_submit_feedback,
    on_submit=None,
    key=f"feedback_{st.session_state.feedback_key}",
    optional_text_label="Please provide some more information",
    kwargs={
        "question": st.session_state["messages"][-2]['content'] if len(st.session_state["messages"]) > 2 else "",
        "response": st.session_state["messages"][-1]['content'] if len(st.session_state["messages"]) > 2 else "",
#        "runtime": st.session_state.runtime,
#        "sources": str(st.session_state.sources),
#        "uuid": str(uuid)
    }
)

#if feedback:
#    send_to_pubsub(feedback)
