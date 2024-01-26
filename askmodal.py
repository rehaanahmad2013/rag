# code taken from vllm modal tutorial 

import os
from modal import Image, Secret, Stub, method

MONGO_PASSWORD = os.getenv('MONGO_PASSWORD')
OPENAI_KEY = os.getenv("OPENAI_KEY")

MODEL_DIR = "/model"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


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
        num_tokens = 0
        for output in result:
            num_tokens += len(output.outputs[0].token_ids)
            print(output.prompt, output.outputs[0].text, "\n\n", sep="")
        print(f"Generated {num_tokens} tokens")

@stub.function(image=langchain_image, secret=Secret.from_name("takehome"))
async def retrievedoc(query: str) -> set[str]:
    uri = "mongodb+srv://rehaan:" + os.environ['MONGO_PASSWORD'] + "@vecdb.ppczayz.mongodb.net/?retryWrites=true&w=majority"
    DB_NAME = "vecdb"
    COLLECTION_NAME = "modalcollect"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    from langchain_community.vectorstores import MongoDBAtlasVectorSearch
    from langchain_openai import OpenAIEmbeddings

    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        uri,
        DB_NAME + "." + COLLECTION_NAME,
        OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ['OPENAI_KEY'], disallowed_special=()),
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    results = vector_search.similarity_search(query)
    return results

# ## Run the model
@stub.local_entrypoint()
def main():
    model = Model()
    questions = "If I want to treat exceptions as successful results and aggregate them in the results list, what do I pass in?"
    result = retrievedoc.remote(questions)

    foundstr = result[0].page_content
    actualstr = result[0].metadata['source']
    getIndex = actualstr.index(foundstr)

    context = actualstr[max(0, getIndex - 200): min(getIndex + 200, len(actualstr))]


    mistralq = [('You are to answer questions about the documentation for a company called Modal Labs. '
              + 'We will provide relevant information from the docs to answer the question. \n'
              + 'Docs: \n'
              + context, 
                questions
              )] 
    
    model.generate.remote(mistralq)
    