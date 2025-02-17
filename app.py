# RAG Q&A Conversation with PDF Inlcuding chat history##
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##set up streamlit
st.title("Conversational RAG with PDF uploads and chathistory")
st.write("Upload Pdf's and chat with their content")

#Input the groq API key
api_key=st.text_input("Enter your GROQ API key:", type="password")


## Check if groq api key is provided
if api_key:
    llm=ChatGroq(groq_api_key=api_key,model_name="qwen-2.5-32b")
    
    #Creating session id
    #chat interface
    session_id=st.text_input("Session ID",value="default_session")
# else:
#     st.write("Please enter your GROQ API key")
#     ## statefully manage chat history
    
    if 'store' not in st.session_state:
        st.session_state.store={} #All the value pair of AI and Humam messages will be stored here
    history=st.session_state.store.get(session_id,ChatMessageHistory())
    uploaded_files=st.file_uploader("Choose a pdf file",type="pdf",accept_multiple_files=True)
    
    ##Process uploaded PDF
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            # temppdf=f"./temp.pdf" #storing in local directory
            temppdf=F"./temp_{uploaded_file.name}"
            with open(temppdf,"wb") as file: 
                file.write(uploaded_file.getvalue()) #Writing the file in local
                file_name=uploaded_file.name #Reading the file name
            loader=PyPDFLoader(temppdf) #Loading the file
            docs=loader.load()
            documents.extend(docs)#appending all docs
            
        #split and create embeddings for documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000,chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings,persist_directory="./chroma_db")
        # vectorstore.persist()
        retriever=vectorstore.as_retriever()
        
        contextualize_q_system_prompt=(
                "Given a chat history and latest user question"
                "which might reference context in the chat history,"
                "formulate a standalone question which can be understood"
                "without the chat history. Do NOT answer the question."
                "just reformulate it if needed and otherwise return it as is."
            ) # The prompt is to reformulate a question without using chat history. It should try to understand its own even if the chat history is lost
            
            
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
                [
                    ("system",contextualize_q_system_prompt),
                    MessagesPlaceholder("chat_history"), #chat history
                    ("human","{input}")
                ]
            )
            
            #Creating chain
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt) #retriever with memory.Formulate a clearer question or search query that makes more sense in context, even if some history is missing.
        
        
        #Answer question prompt
        system_prompt=(
            "You are an assistant for question-answering tasks."
            'Use the following pieces of retrieved context to answer the users question'
            "the question. If you don't know the answer, say that you don't know."
            "Use three sentences maximum and keep the answer concise"
            "\\n\\n"
            "{context}" #This will be replaced bu stuff_document_chain
        )
        
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)#Relevant documents are passed to llm with the help of retriever
        #as context in stuffed document chain and the answer is generated with the help of llm.
        
        #Creating chain
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)# On the basis of the context it will generate answer. 
        #Here becuase of history_aware_retriever the standalone question are formulated and relevant documents 
        # are fetched from database. Because of question_answer_chain it will send the documents to llm in 
        # stuffed document chain and generate answer based on history.
        
        
        #The stuffed document chain (or stuff documents chain) is used to combine multiple retrieved documents 
        # into a single coherent context, which is then used to generate a response to the user's question.
        
        def get_session_history(session:str)->BaseChatMessageHistory: #function is designed to retrieve or create the chat history for a given session.
            if session not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]    
        
        
        conversational_rag_chain=RunnableWithMessageHistory(rag_chain,  #a utility that combines the RAG chain (rag_chain) with message history to make the question-answering process conversational.

                                                            get_session_history,
                                                            input_messages_key="input",
                                                            history_messages_key="chat_history",
                                                            output_messages_key="answer")
    
        # Create a text input field where the user can enter their question
        user_input = st.text_input("Your question:")

        # Check if the user has entered a question (i.e., the input is not empty)
        if user_input:
            
            # Retrieve the session's chat history using the session_id
            session_history = get_session_history(session_id)
            
            # Invoke the conversational retrieval-augmented generation (RAG) chain
            # Pass the user input as the input and the session_id in the config to maintain conversation context
            response = conversational_rag_chain.invoke(
                {"input": user_input},  # User's question as input to the RAG chain
                config={"configurable": {"session_id": session_id}}  # Pass session_id for session-based context
            )
            
            # Display the entire session state store (used for debugging and monitoring state)
            # st.write(st.session_state.store)
            
            # Display the assistant's response (from the RAG chain) to the user
            st.write("Assistant:", response['answer'])
            
            # Show the updated chat history (including the new message and prior conversation)
            st.write("Chat History", session_history.messages)


else:
    st.warning("Please provide your GROQ API key")
    
    
    

