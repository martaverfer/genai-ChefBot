# System libraries
import os
import openai
import argparse

# GenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv

CHROMA_PATH = "../chroma"

PROMPT_TEMPLATE = """
## SYSTEM ROLE
You are a helpful and kind chatbot designed to assist with questions related with cooking.
Your answers must be based exclusively on provided content from books provided. 

## USER QUESTION
The user has asked: 
{question}

## CONTEXT
Here is the relevant content from the technical books:  
'''
{context}
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
2. When explaining a recipe, use bullet points for every ingredient and add the steps
3. When you don't know the answer don't respond with a title format and don't add the source
4. Provide the response in the following format:

## RESPONSE FORMAT

## [Brief Title of the Answer]
[Answer in simple, ordered, clear text.]

**Source**:  
• [Book Title], Page(s): [...]

"""

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    answer = query_rag(query_text)
    print(answer)

def query_rag(query_text: str):
    # Load api_key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") 
    openai.api_key = api_key    

    # Prepare the DB.
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=10)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    # Set up GPT client and parameters
    client = openai.OpenAI()
    model_params = {
        'model': 'gpt-4o',
        'temperature': 0.7,  # Increase creativity
        'max_tokens': 4000,  # Allow for longer responses
        'top_p': 0.9,        # Use nucleus sampling
        'frequency_penalty': 0.5,  # Reduce repetition
        'presence_penalty': 0.6    # Encourage new topics
    }

    messages = [{'role': 'user', 'content': prompt}]
    completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)
    
    return completion.choices[0].message.content

if __name__ == "__main__":
    main()