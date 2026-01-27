import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

## Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv
load_dotenv()

# BACKEND LOGIC (Unchanged)

DB_FAISS_PATH="vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt=PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


# FRONTEND / UI LAYER
def main():
    # --- 1. Page Configuration (Must be first) ---
    st.set_page_config(
        page_title="Enterprise Knowledge Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- 2. Custom CSS for SaaS-like polish ---
    # Hides the default Streamlit menu and footer, adds padding
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stChatInput {padding-bottom: 20px;}
        </style>
    """, unsafe_allow_html=True)

    # --- 3. Sidebar UI ---
    with st.sidebar:
        st.title("🤖 Knowledge Hub")
        st.markdown("---")
        
        st.markdown("### ⚙️ Capabilities")
        st.markdown(
            """
            - 📚 **RAG-Powered:** Searches your internal knowledge base.
            - ⚡ **Groq Llama 3:** High-speed inference.
            - 🧠 **Context Aware:** Understands document nuance.
            """
        )
        
        st.markdown("---")
        
        # Clear Chat Button
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.caption("© 2024 Enterprise AI • v1.0")

    # --- 4. Main Chat Interface ---
    
    # Header Section
    st.markdown("## 💬 Ask the Assistant")
    st.caption("Access the vector database instantly. Ask questions about your uploaded documents.")

    # Initialize session state for chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display Empty State (Welcome Message) if history is empty
    if not st.session_state.messages:
        with st.container():
            st.info("👋 **Welcome!** The database is loaded. Try asking: *'What are the key insights in the documents?'*")

    # Render Chat History
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    # --- 5. User Input & Processing ---
    prompt = st.chat_input("Type your question here...")

    if prompt:
        # Display user message immediately
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role':'user', 'content': prompt})
                
        # Process response with visual feedback (Spinner)
        try: 
            with st.spinner("Analyzing documents..."):
                # --- START BACKEND LOGIC ---
                vectorstore = get_vectorstore()
                if vectorstore is None:
                    st.error("Failed to load the vector store")
                    return

                GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
                GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
                
                llm = ChatGroq(
                    model=GROQ_MODEL_NAME,
                    temperature=0.5,
                    max_tokens=512,
                    api_key=GROQ_API_KEY,
                )
                
                retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

                # Document combiner chain
                combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)

                # Retrieval chain
                rag_chain = create_retrieval_chain(vectorstore.as_retriever(search_kwargs={'k': 3}), combine_docs_chain)

                response = rag_chain.invoke({'input': prompt})
                result = response["answer"]
                # --- END BACKEND LOGIC ---

            # Display Assistant Response
            with st.chat_message('assistant'):
                st.markdown(result)
            
            st.session_state.messages.append({'role':'assistant', 'content': result})

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

if __name__ == "__main__":
    main()