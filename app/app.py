import streamlit as st 
import os  
import openai  
from langchain.embeddings import OpenAIEmbeddings  # Embedding model for vector search
from langchain.vectorstores import Chroma  # Chroma Vector Store for retrieval

# --- Load API key ---
api_key = st.secrets["OPENAI_API_KEY"] 

# ===================================
#              FUNCTIONS
# ===================================

@st.cache_resource
def load_database():
    """Loads the database."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory="../chroma", embedding_function = embeddings)
    return db

def retrieve_docs(user_question):
    """
    Retrieve the documents in ChromaDB
    """
    db = load_database()
    return db.similarity_search(user_question, k=10)

def _get_document_prompt(docs):
    prompt = "\n"
    for doc in docs:
        prompt += "\nContent:\n"
        prompt += doc.page_content + "\n\n"
    return prompt

def prepare_answer(user_question, context_text, source_info):
    """
    Creates an answer based on the prompt
    """
    # --- Construct AI Prompt ---
    prompt = f"""
    ## SYSTEM ROLE
    You are a helpful and kind chatbot designed to assist with questions related with cooking.
    Your answers must be based exclusively on provided content from books provided. 

    ## USER QUESTION
    The user has asked: 
    "{user_question}"

    ## CONTEXT
    Here is the relevant content from the technical books:  
    '''
    {context_text}
    '''

    ## GUIDELINES
    1. **Accuracy**:  
    - Only use the content in the `CONTEXT` section to answer.  
    - If you don't know the answer, just say: Sorry, I cannot help you with that.

    2. **Transparency**:  
    - Reference the book's name and page numbers when providing information.  
    - Do not speculate or provide opinions.  

    3. **Clarity**:  
    - Use simple and concise language.  
    - Format your response in Markdown for readability.  

    ## TASK
    1. Answer the user's question **directly** if possible.  
    2. If the users are asking how to make something, give them the recipe
    3. When explaining a recipe, use bullet points for every ingredient and add the steps
    4. When you don't know the answer don't respond with a title format and don't add the source
    5. Provide the response in the following format:

    ## RESPONSE FORMAT
        
    ## [Brief Title of the Answer]
    [Answer in simple, ordered, clear text.]

     **Source**:  
    {source_info}
        
    """

    # --- Call OpenAI GPT ---
    client = openai.OpenAI()
    model_params = {
        'model': 'gpt-4o', # 
        'temperature': 0.7, # temperature is a hyperparameter that controls the randomness of predictions
        'max_tokens': 4000, # token limit for the response
        'top_p': 0.9,
        'frequency_penalty': 0.5,
        'presence_penalty': 0.6 # presence_penalty is a hyperparameter that penalizes new tokens based on their presence in the context
    }

    messages = [{'role': 'user', 'content': prompt}]
    completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)
    return completion.choices[0].message.content


# ===================================
#              STREAMLIT
# ===================================

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="ChefBot", page_icon="🍳", layout="wide")

# --- Header with Logo & Title ---
st.markdown("""
    <div class='title-container' style='text-align: center;'>
        <h1>🧑🏽‍🍳 ChefBot</h1>
        <p>Your trusted AI-powered chef</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""<br><br>""", unsafe_allow_html=True)  # Spacer

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []
    st.session_state["messages"].append({"role": "assistant", "content": "Hi there! 👋 How can I make your day tastier?"})
    
# --- Display Chat History ---
for message in st.session_state["messages"]:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- User Input ---
user_question = st.chat_input("Ask me anything!")

if user_question: 
    st.session_state["messages"].append({"role": "user", "content": user_question})

    with st.chat_message("user"):
        st.markdown(user_question)

    # --- Processing Animation ---
    with st.spinner("I’m cooking up some ideas for you! 🤔"):

        retrieved_docs = retrieve_docs(user_question)

        if retrieved_docs:
            source_info = "\n".join(
            [f"📖 **Source:** {os.path.basename(doc.metadata.get('source', 'Unknown'))}, Page: {doc.metadata.get('page', 'N/A')}" for doc in retrieved_docs]
            )

            formatted_context = _get_document_prompt(retrieved_docs)
        else:
            source_info = "❌ **No relevant information found.**"
            context_text = "No context available."

        answer = prepare_answer(user_question, formatted_context, source_info)

    # --- Display AI Response ---
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)