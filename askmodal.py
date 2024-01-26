# code taken from vllm modal tutorial 

import os
from modal import Image, Secret, Stub, method

MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
OPENAI_KEY = os.getenv("OPENAI_KEY")

MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
COLLECTION_NAME = "modalcollect_300"

ADDITIONAL_CONTEXT = 200
TOP_K = 3

def download_model_to_folder():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(MODEL_DIR, exist_ok=True)

    snapshot_download(
        BASE_MODEL,
        local_dir=MODEL_DIR,
        token=os.environ["HF_TOKEN"],
    )
    move_cache()

image = (
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04", add_python="3.10"
    )
    .pip_install("vllm==0.2.5", "huggingface_hub==0.19.4", "hf-transfer==0.1.4")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_folder,
        secret=Secret.from_name("huggingface-secret"),
        timeout=60 * 20,
    )
)

langchain_image = Image.debian_slim(
    python_version="3.10"
).pip_install("langchain==0.1.3", "openai==1.9.0", "tiktoken", "pymongo", "certifi", "langchain-openai")


stub = Stub("example-vllm-inference", image=image)

# The `vLLM` library allows the code to remain quite clean.
@stub.cls(gpu="A100", secret=Secret.from_name("huggingface-secret"))
class Model:
    def __enter__(self):
        from vllm import LLM

        # Load the model. Tip: MPT models may require `trust_remote_code=true`.
        self.llm = LLM(MODEL_DIR)
        self.template = """<s>[INST] <<SYS>>
                        {system}
                        <</SYS>>

                        {user} [/INST] """

    @method()
    def generate(self, user_inputs):
        from vllm import SamplingParams

        prompts = [
            self.template.format(system=q[0], user=q[1]) for q in user_inputs
        ]

        sampling_params = SamplingParams(
            temperature=0.75,
            top_p=1,
            max_tokens=800,
            presence_penalty=1.15,
        )

        result = self.llm.generate(prompts, sampling_params)
        for output in result:
            print(output.outputs[0].text)


@stub.function(image=langchain_image, secret=Secret.from_name("takehome"))
async def retrievedoc(query: str) -> set[str]:
    uri = "mongodb+srv://rehaan:" + os.environ['MONGO_PASSWORD'] + "@vecdb.ppczayz.mongodb.net/?retryWrites=true&w=majority"
    DB_NAME = "vecdb"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    from langchain_community.vectorstores import MongoDBAtlasVectorSearch
    from langchain_openai import OpenAIEmbeddings

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        uri,
        DB_NAME + "." + COLLECTION_NAME,
        OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ['OPENAI_KEY'], disallowed_special=()),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    results = vector_search.similarity_search(query, k=TOP_K)
    return results

# ## Run the model
@stub.local_entrypoint()
def main():
    model = Model()
    # questions = "If I want to treat exceptions as successful results and aggregate them in the results list, what do I pass in?"
    questions = "How do I use modal serve to serve a web endpoint?"
    results = retrievedoc.remote(questions)
    contextStr = ""
    
    for i in range(0, len(results)):
        foundstr = results[i].page_content
        actualstr = results[i].metadata['source']
        getIndex = actualstr.index(foundstr)
        context = actualstr[max(0, getIndex - ADDITIONAL_CONTEXT): min(getIndex + ADDITIONAL_CONTEXT, len(actualstr))]
        contextStr += (context + "\n")

    mistralq = [('You are to answer a user question about the documentation for Modal Labs, a company that helps people run code in the cloud. '
              + 'We will provide sections of Modal Labs documentation to help you answer the question. Note that not all of this information may be relevant for answering the question. \n'
              + 'Documentation: \n'
              + context, 
                questions
              )] 
    
    model.generate.remote(mistralq)

    print("Some relevant links: ")
    for i in range(0, len(results)):
        print("[" + str(i) + "] " + results[i].metadata['link'])

    