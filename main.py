import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import sys
print(sys.version)
import langchain
print(langchain.__version__)
from langchain.globals import set_debug
set_debug(True)

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document if file is uploaded
    if uploaded_file is not None:
        raw_documents = [uploaded_file.read().decode()]
        print(raw_documents)

        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len
        )
        documents = text_splitter.create_documents(raw_documents)
        # Debug : 
        print(f"Found {len(documents)} documents!!")

        # Select embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create a vectorstore from documents
        vectorstore = FAISS.from_documents(documents, embeddings)

        # Create retriever interface
        retriever = vectorstore.as_retriever()

        # Create LLM
        llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4", temperature=0)

        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # create Prompt from query text
        
        template = """You are an AI assistant for answering questions about the most recent state of the union address.
        You are given the following extracted parts of a long document and a question. Provide a conversational answer.
        If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
        If the question is not about the most recent state of the union, politely inform them that you are tuned to only answer questions about the most recent state of the union.
        Question: {question}
        =========
        {context}
        =========
        Answer in Markdown:"""

        QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["question", "context"]
        )

        # Create ConversationalRetrievalChain
        qa = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT}
        )

        return qa({"question": query_text})
    
# Page title
st.set_page_config(page_title='Ask the Transavia FAQ')
st.title("Ask the Transavia FAQ")

# File Upload
uploaded_file = st.file_uploader("Upload an article", type='txt')
# Query text
query_text = st.text_input("Enter your question:", placeholder="Please provide a short summary.", disabled= not uploaded_file)

# Form input and query
result = []
with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input("OpenAI API Key", type="password", disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button("Submit", disabled=not (uploaded_file and query_text))
    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner("Calculating..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
            del openai_api_key

if len(result):
    st.info(response)
