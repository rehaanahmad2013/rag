# # Fast inference with vLLM (Mistral 7B)
#
# In this example, we show how to run basic inference, using [`vLLM`](https://github.com/vllm-project/vllm)
# to take advantage of PagedAttention, which speeds up sequential inferences with optimized key-value caching.
#
# `vLLM` also supports a use case as a FastAPI server which we will explore in a future guide. This example
# walks through setting up an environment that works with `vLLM ` for basic inference.
#
# We are running the [Mistral 7B Instruct](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) model here, which is an instruct fine-tuned version of Mistral's 7B model best fit for conversation.
# You can expect 20 second cold starts and well over 100 tokens/second. The larger the batch of prompts, the higher the throughput.
# For example, with the 60 prompts below, we can produce 19k tokens in 15 seconds, which is around 1.25k tokens/second.
#
# To run
# [any of the other supported models](https://vllm.readthedocs.io/en/latest/models/supported_models.html),
# simply replace the model name in the download step. You may also need to enable `trust_remote_code` for MPT models (see comment below)..
#
# ## Setup
#
# First we import the components we need from `modal`.

import os
# from dotenv import load_dotenv
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
    # Use the barebones hf-transfer package for maximum download speeds. No progress bar, but expect 700MB/s.
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
    def generate(self, user_questions):
        from vllm import SamplingParams

        prompts = [
            self.template.format(system="", user=q) for q in user_questions
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
# We define a [`local_entrypoint`](/docs/guide/apps#entrypoints-for-ephemeral-apps) to call our remote function
# sequentially for a list of inputs. You can run this locally with `modal run vllm_inference.py`.
@stub.local_entrypoint()
def main():
    model = Model()
    questions = [
        # Coding questions
        "If I want to treat exceptions as successful results and aggregate them in the results list, what do I pass in?",
    ]

    result = retrievedoc.remote(questions[0])

    foundstr = result[0].page_content
    actualstr = result[0].metadata['source']
    getIndex = actualstr.index(foundstr)

    context = actualstr[max(0, getIndex - 200): min(getIndex + 200, len(actualstr))]


    mistralq = ['You are to answer questions about the documentation for a company called Modal Labs. '
              + 'We will provide relevant information from the docs to answer the question. \n'
              + 'Docs: \n'
              + context 
              + '\n Question: '
              + questions[0]]
    
    model.generate.remote(mistralq)
    