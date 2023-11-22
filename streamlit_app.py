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
#from source_retriever import list_top_k_sources, get_top_k_urls
#logging.basicConfig(level=logging.INFO)

#uuid = uuid.uuid4()

#aiplatform.init(
#    project=PROJECT_ID,
#    location=REGION,
#    staging_bucket=f"gs://{BUCKET_NAME}"
#)

def get_qa_agent():
    return lambda x: "this is an agent"

@st.cache_resource
def cached_qa_agent():
    qa_agent = get_qa_agent()
    return qa_agent


st.title('Chat with Transavia FAQ')

qa_agent = cached_qa_agent()
#octo_logo = add_logo(logo_path="./logo_octo.png", width=250, height=120)

with st.sidebar:
#    st.image(octo_logo)
    st.caption('Select the documents you want to search in:')
    luggage_docs = st.checkbox('luggage docs', value=True)
    if luggage_docs:
        pass

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
    st.session_state["messages"] = [{"role": "assistant", "content": "Hello, I'm the Transavia Assistant, how can I help you ?"}]

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
    result = qa_agent({"query": prompt})
#    sources_str = list_top_k_sources(result)
#    st.session_state.sources = get_top_k_urls(result)
#    answer = result["result"]
    answer = result
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