# Training on documents
# Embedding and vectors stores 

#pip install --upgrade langchain
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

from langchain.chains import RetrievalQA # retrieve over documents
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader # load proprietary data
from langchain.vectorstores import DocArrayInMemorySearch # in-memory, makes it easy to get i started
from IPython.display import display, Markdown

file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)

# Allows creation of vector store easily
from langchain.indexes import VectorstoreIndexCreator

#pip install docarray
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

response = index.query(query)

display(Markdown(response))

loader = CSVLoader(file_path=file)

docs = loader.load()

# We don't need to chunk if things are small
docs[0]

from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

embed = embeddings.embed_query("Hi my name is Harrison")

 
print(len(embed))

print(embed[:5])

# Creating vector store (in-memory)
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

query = "Please suggest a shirt with sunblocking"

docs = db.similarity_search(query)

len(docs)

docs[0]

# Interface for fetching documents
retriever = db.as_retriever()

llm = ChatOpenAI(temperature = 0.0)

qdocs = "".join([docs[i].page_content for i in range(len(docs))])

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") 

# Does Retrieval, then does QA over documents
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff",  # stuffs into documents into context
    retriever=retriever, 
    verbose=True
)

query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."

response = qa_stuff.run(query)

display(Markdown(response))

response = index.query(query, llm=llm)

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch,
    embedding=embeddings,
).from_loaders([loader])