# Agentic Wikipedia Proof-of-Concept

## Data Source

The Wikipedia Loader ingests documents from the Wikipedia API and converts them into LangChain document objects. The page content includes the first sections of the Wikipedia articles and the metadata is described in detail below.

__Recommendation__: If you are using the LangChain document loader we recommend filtering down to 10k or fewer documents. The `query_terms` argument below can be updated to update the search term used to search wikipedia. Make sure you update this based on the use case you defined.

In the metadata of the LangChain document object; we have the following information:

| Column  | Definition                                                                 |
|---------|-----------------------------------------------------------------------------|
| title   | The Wikipedia page title (e.g., "Quantum Computing").                       |
| summary | A short extract or condensed description from the page content.             |
| source  | The URL link to the original Wikipedia article.                             |


```python
%pip install -U -qqqq backoff databricks-langchain langgraph==0.5.3 uv databricks-agents mlflow-skinny[databricks] chromadb sentence-transformers langchain-huggingface langchain-chroma wikipedia faiss-cpu
dbutils.library.restartPython()
```

    [43mNote: you may need to restart the kernel using %restart_python or dbutils.library.restartPython() to use updated packages.[0m



```python
 #######################################################################################################
 ###### Python Package Imports for this notebook                                                  ######
 #######################################################################################################

from langchain.document_loaders import WikipediaLoader
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
# from langchain.embeddings import DatabricksEmbeddings

from databricks_langchain import (
    ChatDatabricks,
    DatabricksEmbeddings,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)

 
 #######################################################################################################
 ###### Config (Define LLMs, Embeddings, Vector Store, Data Loader specs)                         ######
 #######################################################################################################

# DataLoader Config
query_terms = ["sport", "football", "soccer", "basketball","baseball", "track","swimming", "gymnastics"] #TODO: update to match your use case requirements
max_docs = 10 #TODO: recommend starting with a smaller number for testing purposes

# Retriever Config
k = 2 # number of documents to return
EMBEDDING_MODEL = "databricks-bge-large-en" # Embedding model endpoint name


# LLM Config
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-1-8b-instruct" # Model Serving endpoint name; other option see "Serving" under AI/ML tab (e.g. databricks-gpt-oss-20b)


example_question = "What is the most popular sport in the US?"

```


```python
 #######################################################################################################
 ###### Wikipedia Data Loader                                                                     ######
 #######################################################################################################

docs = WikipediaLoader(query=query_terms, load_max_docs=max_docs).load() # Load in documents from Wikipedia takes about 10 minutes for 1K articles

#######################################################################################################
###### FAISS Retriever: Using DBX embedding model                                                ###### #######################################################################################################

# Define the embeddings and the FAISS vector store
embeddings = DatabricksEmbeddings(endpoint=EMBEDDING_MODEL) # Use to generate embeddings
vector_store = FAISS.from_documents(docs, embeddings)
 
# Example of how to invoke the vector store
results = vector_store.similarity_search(
    "What is the most popular sport in the US?",
    k=k
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")

#######################################################################################################
###### LLM: Using DBX Foundation Model                                                           ###### #######################################################################################################

llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

response = llm.invoke("What is the most popular sport in the US?")

print("\n",response.content)
```
---
    * The USA Gymnastics National Championships is the annual artistic gymnastics national competition held in the United States for elite-level competition. It is currently organized by USA Gymnastics, the governing body for gymnastics in the United States. The national championships have been held since 1963.
---

### a) GenAI Application Development

__REQUIRED__: This section is where input your custom logic to create and run your agentic workflow. Feel free to add as many codes cells that are needed for this assignment


```python
#TODO: Enter your Agentic workflow code here
```

### b) Reflection

__REQUIRED:__ Provide a detailed reflection addressing  these two questions:
1. If you had more time, which specific improvements or enhancements would you make to your agentic workflow, and why?
2. What concrete steps are required to move this workflow from prototype to production?


> Enter your reflection here



### 
