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
