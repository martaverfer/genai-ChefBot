# 🧑🏽‍🍳 ChefBot: Your trusted AI-powered chef

ChefBot is an **AI-powered chatbot** designed to assist with cooking-related queries. Whether you're looking for recipe ideas, cooking techniques, or ingredient substitutions, ChefBot is here to help! This project leverages large language model (LLM) to generate responses based on retrieved documents to deliver intelligent, context-aware responses.

## Features

- **Recipe Suggestions:** Get personalized recipe ideas based on available ingredients.

- **Dietary Restrictions:** ChefBot can filter recipes and suggestions based on dietary needs (e.g., vegetarian, gluten-free, vegan).

- **Basic Cooking Concepts:** this AI-powered chef can help you understand essential cooking techniques and terminology.

- **Cooking and Grocery Tips:** ChefBot can offer advice on food storage, grocery shopping, meal planning, and tips to improve your kitchen efficiency

- **Seasonal Food:** Get recommendations based on what ingredients are in season, helping you make the best of what's available.

## How it works
ChefBot uses several advanced techniques to generate its responses. Below is a breakdown of the key steps involved in the chatbot's operation:

- **Loading Data:** The data is essential to how ChefBot generates its responses based on the PDF loaded.

- **Chunking the Data**: Once the data is loaded, we split it into smaller chunks based on logical separations in the data. This makes it easier to search and retrieve specific sections of the content.

- **Embedding Data Using OpenAI:** To make the chunks searchable and retrievable by the chatbot, we convert the chunks into embeddings. Embeddings are numerical representations of the text that capture its meaning in a way that a machine can understand.

- **Connecting to the Large Language Model (LLM):** After embedding the data, ChefBot uses a Large Language Model (LLM) to generate the final response based on user queries. The LLM is connected to the vector database (ChromaDB) to retrieve relevant chunks, and it combines these chunks with the query to generate an accurate and contextual response. In our case, we use OpenAI’s GPT-4 model to generate responses.


## 💻 Installation: Getting Started

To run this project locally, follow these instructions:

### 1. **Clone the repo:**

```bash
git clone https://github.com/martaverfer/genai-ChefBot; \
cd python_scripts
```

### 2. **Virtual environment:**

Create the virtual environment: 
```bash
python3 -m venv venv
```

Activate the virtual environment:

- For Windows: 
```bash
venv\Scripts\activate
```

- For Linux/Mac: 
```bash
source venv/bin/activate
```

To switch back to normal terminal usage or activate another virtual environment to work with another project run:
```deactivate```

### 3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

### 4. **Run the python scripts to explore the chatbot:**
```bash
cd python_scripts; 
populate_db.py
query_data.py
```

### 5. **Evaluation and Testing:**
ChefBot is evaluated using various queries to ensure it responds accurately and provides useful information. The testing framework used is pytest, and you can run the tests with:
``` bash
cd python_scripts; 
python -m pytest -s
```
This will run all the tests and print output to the terminal, allowing you to verify that everything is working as expected.

### 6. **Running Streamlit App**
```bash
cd app; 
streamlit run app.py
```