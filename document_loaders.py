# URL document loader

from langchain.document_loaders import WebBaseLoader

loader = WebBaseLoader('http://www.kdpress.co.kr/news/articleView.html?idxno=120042')
data = loader.load()
print(data[0].page_content)

# 여러개 url을 한번에 불러오기
from langchain.document_loaders import UnstructuredURLLoader

urls = [
    'http://www.kdpress.co.kr/news/articleView.html?idxno=120042',
    'https://www.chosun.com/national/weekend/2023/06/03/YUFW474WWFHNNCKCN2YOWXNTKI/'
]

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
data

# PDF loader
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("/home/blakej/langchain_tutorial/딜로이트_미래를 결정해야 하는 순간 생성형AI 도입 현장에서 확인한 시사점.pdf")
pages = loader.load_and_split()
pages[0]

pages[1].page_content

from langchain.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("/content/drive/MyDrive/langchain_연습/해커톤.docx")
data = loader.load_and_split()
data[1].metadata
print(data[0].page_content)
print(type(data[0].page_content))


# chroma


import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

tokenizer = tiktoken.get_encoding('cl100k_base') # openai에서 사용하는 토큰세는 라이브러리

def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader


# load the document and split it into chunks
loader = PyPDFLoader("/content/drive/MyDrive/공고문(2024+서울+열린데이터광장+공공데이터+활용+창업경진대회).pdf")
pages = loader.load_and_split()

# split it into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, length_function = tiktoken_len)
docs = text_splitter.split_documents(pages)

# create the open-source embedding function
from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

# load it into Chroma
db = Chroma.from_documents(docs, hf)

# query it
query = "발전가능성에 대해서?"
docs = db.similarity_search(query)

# print results
print(docs[0].page_content)  # 가장 유사한 문장을 뽑고 위에서 정한 chunkP_size에 따라 길이가 결정됨

tiktoken_len(docs[0].page_content)

# save to disk
db2 = Chroma.from_documents(docs, hf, persist_directory="./chroma_db")
docs = db2.similarity_search(query)

# load from disk
db3 = Chroma(persist_directory="./chroma_db", embedding_function=hf)
docs = db3.similarity_search(query)
print(docs[3].page_content)


docs = db3.similarity_search_with_relevance_scores(query, k=4) # 유사도 높은 문서 중 저장할 개수


print("가장 유사한 문서:\n\n {}\n\n".format(docs[3][0].page_content))
print("문서 유사도:\n {}".format(docs[3][1]))



# 크로마 db에 넣은 임베딩 확인하는 법
# db3.get(ids="08e843be-e0c9-4991-83d5-112689fb43a9", include=["embeddings"])



