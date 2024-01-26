from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi 
uri = "mongodb+srv://rehaan:BK9MEnBVz59CZbkB@vecdb.ppczayz.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
client = MongoClient(uri, tlsCAFile=certifi.where())

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


DB_NAME = "vecdb"
COLLECTION_NAME = "modalcollect"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
data = loader.load()

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(data)

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings.openai import OpenAIEmbeddings

# insert the documents in MongoDB Atlas with their embedding
vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="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", disallowed_special=()),
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

# Perform a similarity search between the embedding of the query and the embeddings of the documents
query = "What were the compute requirements for training GPT 4"
results = vector_search.similarity_search(query)

print(results[0].page_content)
