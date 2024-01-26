from dotenv import load_dotenv
load_dotenv()
from itertools import zip_longest
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import streamlit as st
from streamlit_chat import message
from langchain.callbacks.base import BaseCallbackHandler
import openai
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
import tempfile


#제목
# Set streamlit page configuration
st.set_page_config(page_title="CSV파일 기반 챗봇")
st.title("ChatBot Starter")


#OpenAI KEY 입력 받기
# openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

#파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 올려주세요!",type=['csv'])
st.write("---")

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input

# Initialize the ChatOpenAI model
chat = ChatOpenAI(
    temperature=0.5,
    model_name="gpt-3.5-turbo"
)

def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        content="You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer.")]

    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state['past'], st.session_state['generated']):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages


def generate_response():
    """
    Generate AI response using the ChatOpenAI model.
    """
    # Build the list of messages
    zipped_messages = build_message_list()

    # Generate response using the chat model
    ai_response = chat(zipped_messages)

    return ai_response.content


# Define function to submit user input
def submit():
    # Set entered_prompt to the current value of prompt_input
    st.session_state.entered_prompt = st.session_state.prompt_input
    # Clear prompt_input
    st.session_state.prompt_input = ""

# Create a text input for user
st.text_input('YOU: ', key='prompt_input', on_change=submit)    

if st.session_state.entered_prompt != "":
    # Get user query
    user_query = st.session_state.entered_prompt

    # Append user query to past queries
    st.session_state.past.append(user_query)

    # Generate response
    output = generate_response()

    # Append AI response to generated responses
    st.session_state.generated.append(output)

# Display the chat history
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        # Display AI response
        message(st.session_state["generated"][i], key=str(i))
        # Display user message
        message(st.session_state['past'][i],
                is_user=True, key=str(i) + '_user')

# #업로드 되면 동작하는 코드
if uploaded_file is not None:

   #use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="euc-kr", csv_args={
                'delimiter': ','})
    data = loader.load()
  
    # st.write(data)

    #Split
    # text_splitter = RecursiveCharacterTextSplitter(
    #     # Set a really small chunk size, just to show.
    #     chunk_size = 300,
    #     chunk_overlap  = 20,
    #     length_function = len,
    #     is_separator_regex = False,
    # )
    # texts = text_splitter.split_documents(pages)

    #Embedding
    embeddings_model = OpenAIEmbeddings()
    vectorstores = FAISS.from_documents(data, embeddings_model)
    #persist_directory
    # persist_directory="C:\langchain/chatpdf2/"

    # load it into Chroma
    # db = Chroma.from_documents(texts, embeddings_model)
    # db = Chroma.from_documents(data, embeddings_model, persist_directory=persist_directory)
    # db.persist()

    #Stream 받아 줄 Hander 만들기(대답 자연스럽게)
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""):
            self.container = container
            self.text=initial_text
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            self.text+=token
            self.container.markdown(self.text)

    #Question
    st.header("CSV에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')

   
    if st.button('질문하기'):
        with st.spinner('Wait for it...'):

            chat_box = st.empty()
            stream_hander = StreamHandler(chat_box)
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True, callbacks=[stream_hander])
            qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectorstores.as_retriever())
            qa_chain({"query": question})
	