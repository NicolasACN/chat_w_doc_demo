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
from pydantic.v1 import BaseModel, Field
from typing import Optional, List
from langchain.tools import tool
import sqlite3
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor
from langchain.prompts import MessagesPlaceholder
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents.format_scratchpad import format_to_openai_functions
from langchain.tools.render import format_tool_to_openai_function
from langchain.memory import ConversationBufferMemory


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

def extract_chunk_store(folder_path: str):
    # Extract 
    loader = DirectoryLoader(folder_path, glob="*.txt")
    documents = loader.load()

    # Chunk
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separator="\n\n")
    chunks = text_splitter.split_documents(documents)
    
    # Store
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db

db = extract_chunk_store(folder_path="./FAQ/")
retriever = lambda x: db.similarity_search(query=x, k=5, return_metadata=True)

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

class SearchFAQInput(BaseModel):
    query: str = Field(description="Optimal user query to search the Transavia FAQ for")
    #filter: Optional[MetadataFilterEnum] = Field(description="Filter extracted from the user query to search the specific section of the FAQ", )
    
    
@tool(args_schema=SearchFAQInput)
def search_FAQ(query: str) -> str:
    """Searches Transavia FAQ with query extracted from user messages and using the appriate filter if identified

    Args:
        query (str): _description_

    Returns:
        str: Identified fragment of the text relevant to answer the user's query question
    """
    return retriever(query)

class Passenger(BaseModel):
    """Information about the passenger"""
    name: str = Field(description="first name of the passenger")
    surname: str = Field(description="surname of the passenger")
    passport_number: str = Field(description="passport number of the passenger")
    
class Flight(BaseModel):
    """Information about the flight"""
    flight_id: str = Field(description="Unique identifier number for the flight")
    flight_status: str = Field(description="Status of the flight, On Time or Delayed")
    origin: str = Field(description="Origin airport of the flight")
    destination: str = Field(description="destination airport of the flight")
    seats_available: int = Field(description="Number of seats currently available on the flight")
    departure_date: str = Field(description="Flight date of departure")
    departure_time: str = Field(description="Time of departure of the flight")

class BookTripInput(BaseModel):
    passenger_list: List[Passenger] = Field(description="list of the passengers traveling")
    departure_airport: str = Field(description="departure airport code")
    destination_airport: str = Field(description="destination airport code")
    trip_date: str = Field(description="date of the trip, in YYYY-MM-DD format")

import random

@tool(args_schema=BookTripInput)
def book_trip(passenger_list: List[Passenger], departure_airport: str, destination_airport: str, trip_date: str) -> str:
    """Book a transavia flight for the list of passengers"""
    if not departure_airport:
        return "Trip not booked. Departure Airport info is missing."
    if not destination_airport:
        return "Trip not booked. Destination Airport info is missing"
    if not trip_date:
        return "Trip not booked. Trip Date info is midding"
    for passenger in passenger_list:
        if not passenger["name"]: 
            return "Trip not booked. Passenger Name info is missing"
        if not passenger['surname']:
            return "Trip not booked. Passenger Surname info is missing"
        if not passenger["passport_number"]:
            return "Trip not booked. Passenger Passport Number info is missing"
    
    conn = sqlite3.connect("../db/transavia_demo.db")
    cursor = conn.cursor()
    
    for passenger in passenger_list:
        insert_query = """
        INSERT INTO bookings (booking_id, name, surname, origin, destination, departure_date, flight_status, passport_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.execute(insert_query, (random.randint(10**4, 10**6), passenger["name"], passenger["surname"], departure_airport, destination_airport, trip_date, "ON TIME", passenger["passport_number"])) # TODO : status hardcoded, this is wrong
        conn.commit()
        conn.close()
    
    return "Trip booked!"



import sqlite3
from datetime import datetime, timedelta

class SearchFlightsInput(BaseModel):
    origin: str = Field(description="origin airport code for the flight")
    destination: str = Field(description="destination airport code for the flight")
    date: str = Field(description="Date for the flight in the form YYYY-MM-DD")

@tool(args_schema=SearchFlightsInput)
def list_flights(origin, destination, date=None):
    """List transavia flights according to what the user is looking for"""
    conn = sqlite3.connect("../db/transavia_demo.db")
    cursor = conn.cursor()
    
    if not date:
        query = """
        select * from flights
        where origin = ?
        and destination = ?
        """
        answer = cursor.execute(query, (origin, destination)).fetchall()
    
    else:
        if date == "today":
            date = datetime.now().strftime("%Y-%m-%d")
        elif date == "tomorrow":
            today = datetime.now()
            tomorrow = today + timedelta(days=1)
            date = tomorrow.strftime("%Y-%m-%d")
            
        query = """
        select * from flights
        where origin = ?
        and destination = ?
        and departure_date >= ?
        """
        answer = cursor.execute(query, (origin, destination, date)).fetchall()
    
    conn.close()
    
    return answer
        
class SearchBookingsInput(BaseModel):
    booking_id: Optional[str] = Field(description="Booking ID of the user booking")
    passenger_info: Passenger = Field(description="Passenger information including name, surname and passeport ID (optional)")
    
@tool(args_schema=SearchBookingsInput)
def search_bookings(passenger_info: Passenger, booking_id: int = None):
    """Search a booking for a Transavia custommer. User must provide at least name, surname and passport_id or booking_id"""
    conn = sqlite3.connect("../db/transavia_demo.db")
    cursor = conn.cursor()
    
    if not passenger_info['passport_number'] and not booking_id:
        return "Please provide at least the ID of the booking or the passport number of the passenger"
        
    if not passenger_info['passport_number']:
        query = """
        SELECT * FROM bookings
        WHERE booking_id = ?
        AND name = ?
        AND surname = ?
        """
        answer = cursor.execute(query, (booking_id, passenger_info['name'], passenger_info["surname"])).fetchall()
    
    elif not booking_id:
        query = """
        SELECT * FROM bookings
        WHERE name = ?
        AND surname = ?
        AND passport_id = ?
        """
        answer = cursor.execute(query, (passenger_info["name"], passenger_info["surname"], passenger_info["passport_number"])).fetchall()
    
    else:
        query = """
        SELECT * FROM bookings
        WHERE booking_id = ?
        AND name = ?
        AND surname = ?
        AND passport_id = ?
        """
        answer = cursor.execute(query, (booking_id, passenger_info["name"], passenger_info["surname"], passenger_info["passport_number"])).fetchall()
    
    conn.close()
    
    return answer
        
def get_agent():
    # Initialize VDB and retriever
    db = extract_chunk_store(folder_path="../FAQ/")
    retriever = lambda x: db.similarity_search(query=x, k=5, return_metadata=True)
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful customer assistant working for the Transavia company. Great them and assist them as best as you can. You use the tools at your disposal to satisfy user needs. If you can't help the user with its query, ask them to email the support at support@transavia.com."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    tools = [search_FAQ, book_trip, list_flights, search_bookings]
    model = ChatOpenAI(temperature=0).bind(functions=[format_tool_to_openai_function(t) for t in tools])
    agent_chain = RunnablePassthrough.assign(
        agent_scratchpad= lambda x: format_to_openai_functions(x["intermediate_steps"])
    ) | prompt | model | OpenAIFunctionsAgentOutputParser()
    memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")
    agent_executor = AgentExecutor(memory=memory, agent=agent_chain, tools=tools, verbose=True)
        
    return agent_executor

@st.cache_resource
def cached_agent():
    agent = get_agent()
    return agent

# Logo
scaling = 0.2
transavia_logo = add_logo(logo_path="./logo/Transavia_logo.svg.png", width=int(scaling*1280), height=int(scaling*250))
st.image(transavia_logo)


st.title("Transavia GenAI Assistant")

qa_agent = cached_agent()



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
