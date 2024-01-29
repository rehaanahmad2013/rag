import os

import modal

stub = modal.Stub("example-linkscraper")

langchain_image = modal.Image.debian_slim(
    python_version="3.10"
).pip_install("langchain==0.1.3", "openai==1.9.0", "tiktoken", "pymongo", "certifi", "langchain-openai")

playwright_image = modal.Image.debian_slim(
    python_version="3.10"
).run_commands(  # Doesn't work with 3.11 yet
    "apt-get update",
    "apt-get install -y software-properties-common",
    "apt-add-repository non-free",
    "apt-add-repository contrib",
    "pip install playwright==1.30.0",
    "playwright install-deps chromium",
    "playwright install chromium",
)

CHUNK_SIZE = 500

@stub.function(image=playwright_image)
async def list_links(url: str) -> set[str]:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(url)
        links = await page.eval_on_selector_all(
            "a[href]", "elements => elements.map(element => element.href)"
        )
        
        await browser.close()

    return set(links)

@stub.function(image=playwright_image)
async def get_content(url: str) -> set[str]:
    from playwright.async_api import async_playwright
    
    html = ""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.goto(url)
            html = await page.locator('article').text_content()

            await browser.close()
    except:
        html = ""

    if len(html) > 0:
        uploadpage.remote((html, url))
    
    return True

@stub.function()
def scrape():
    links_of_interest = ["https://modal.com/docs/examples", "https://modal.com/docs/guide", "https://modal.com/docs/reference"]

    returnLinks = []
    for links in list_links.map(links_of_interest):
        for link in links:
            if "modal.com/docs/" in link:
                if link not in returnLinks:
                    returnLinks.append(link)

    return returnLinks

@stub.function(image=langchain_image, secret=modal.Secret.from_name("takehome"))
async def uploadpage(html: tuple) -> set[str]:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings
    from pymongo.mongo_client import MongoClient
    import certifi 
    from langchain_community.vectorstores import MongoDBAtlasVectorSearch

    uri = "mongodb+srv://rehaan:" + os.environ['MONGO_PASSWORD'] + "@vecdb.ppczayz.mongodb.net/?retryWrites=true&w=majority"
    client = MongoClient(uri, tlsCAFile=certifi.where())

    DB_NAME = "vecdb"
    COLLECTION_NAME = "modalcollect_" + str(CHUNK_SIZE)
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"
    MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]

    chunk_overlap = 50
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    chunks = text_splitter.create_documents(
        texts=[html[0]],
        metadatas=[{"source": html[0], "link": html[1]}])

    MongoDBAtlasVectorSearch.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=os.environ['OPENAI_KEY'], disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    return True


@stub.local_entrypoint()
def run():
    docLinks = scrape.remote()

    for html in get_content.map(docLinks):
        pass


# @stub.local_entrypoint()
# def run():
#     docLinks = scrape.remote()

#     html_arr = []
#     for html in get_content.map(docLinks):
#         pass
#         if len(html[0]) > 0:
#             html_arr.append(html)

#     for x in uploadpage.map(html_arr):
#         pass