import time
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from footer import footer
from openai import OpenAI

# Set the Streamlit page configuration and theme
st.set_page_config(page_title="Legal Advice System", layout="centered")

# Display the logo image
col1, col2, col3 = st.columns([1, 30, 1])
with col2:
    st.image("images/supreme-court.jpeg", use_container_width=True)

def hide_hamburger_menu():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)

hide_hamburger_menu()

# Initialize session state for messages and memory
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

@st.cache_resource
def load_embeddings():
    """Load and cache the embeddings model."""
    return HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")

embeddings = load_embeddings()
db = FAISS.load_local("legal_embed_db", embeddings, allow_dangerous_deserialization=True)
db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3, "filter": lambda doc: "IPC" in doc['source'] or "Constitution" in doc['source']})

prompt_template = """
<s>[INST]
You are a legal chatbot specializing in Indian law. Answer user queries strictly based on the Indian Penal Code (IPC) and the Constitution of India. Ensure:
- Responses are concise, clear, and legally accurate.
- Avoid referencing any foreign laws or unrelated legal systems.
- Provide the section or article number from IPC or the Constitution when applicable.
- If a question is outside the scope of Indian law, inform the user and suggest contacting an expert.

CONTEXT: {context}
CHAT HISTORY: {chat_history}
QUESTION: {question}
ANSWER:
- [Detail the first key aspect, citing relevant sections or articles]
- [Provide clear explanations relevant to Indian law only]
- [Mention exceptions or caveats if applicable]
- [Conclude with any additional relevant legal points]
</s>[INST]
"""

prompt = PromptTemplate(template=prompt_template,
                        input_variables=['context', 'question', 'chat_history'])

# Initialize NVIDIA API client
api_key = os.getenv('NVIDIA_API_KEY')
if not api_key:
    raise ValueError("NVIDIA_API_KEY environment variable is not set")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key="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"
)

def query_nvidia_model(prompt):
    """Query the NVIDIA model."""
    completion = client.chat.completions.create(
        model="mistralai/mixtral-8x22b-instruct-v0.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=1024,
        stream=True
    )
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            response += chunk.choices[0].delta.content
    return response

def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

def is_indian_law_query(query):
    """Check if the query is related to Indian law."""
    keywords = ['ipc', 'indian penal code', 'constitution of india', 'article', 'section']
    return any(keyword in query.lower() for keyword in keywords)

# Chat interaction logic
input_prompt = st.chat_input("Ask your legal question here...")
if input_prompt:
    if not is_indian_law_query(input_prompt):
        st.warning("Unable to find anser, please provide more context.")
    else:
        with st.chat_message("user"):
            st.markdown(f"**You:** {input_prompt}")
        st.session_state.messages.append({"role": "user", "content": input_prompt})

        # Query NVIDIA model and display response
        with st.chat_message("assistant"):
            with st.spinner("Thinking 💡..."):
                context = ""
                chat_history = [
                    {"role": message["role"], "content": message["content"]}
                    for message in st.session_state.memory.chat_memory.messages
                ]
                full_prompt = f"""
                <s>[INST]
                CONTEXT: {context}
                CHAT HISTORY: {chat_history}
                QUESTION: {input_prompt}
                </s>[INST]
                """
                result = query_nvidia_model(full_prompt)

                message_placeholder = st.empty()
                full_response = ""
                for chunk in result:
                    full_response += chunk
                    time.sleep(0.02)
                    message_placeholder.markdown(full_response + " |", unsafe_allow_html=True)

            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Button to reset the chat
if st.button('🗑️ Reset All Chat', on_click=reset_conversation):
    st.experimental_rerun()

footer()
