from pymongo.mongo_client import MongoClient
uri = "mongodb+srv://rehaan:<password>@vecdb.ppczayz.mongodb.net/?retryWrites=true&w=majority"

# Create a new client and connect to the server
# client = MongoClient(uri, tlsCAFile=certifi.where())
client = MongoClient(uri)

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


DB_NAME = "vecdb"
COLLECTION_NAME = "modalcollect"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    uri,
    DB_NAME + "." + COLLECTION_NAME,
    OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key="<KEY-HERE>", disallowed_special=()),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)

# Perform a similarity search between the embedding of the query and the embeddings of the documents
query = "What is the easiest way to increase performance when running the same function repeatedly?"
# query = "What do I do if I'm tired of prompt engineering?"
results = vector_search.similarity_search(query)
# print(results)
print(results[0].page_content)
print(results[0].metadata['source'])

str1 = results[0].page_content
str2 = results[0].metadata['source']

breakpoint()