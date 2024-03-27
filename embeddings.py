
# 한국어 임베딩 성능 좋은 모델 jhgan/ko-sroberta-multitask, ko-sbert-nli, KosimSCE-roberta-Multitask

from langchain.embeddings import OpenAIEmbeddings

embed_model = OpenAIEmbeddings(
    openai_api_key='',
    model = 'text-embedding-3-small'
)

embed_model

embeddings = embed_model.embed_documents(
    [
        "안녕하세요",
        "제 이름은 홍길동입니다.",
        "이름이 무엇인가요?",
        "랭체인은 유용합니다"
    ]
)

embedded_query_q = embed_model.embed_query('이 대화에서 언급된 이름은 무엇입니까?')

embedded_query_a = embed_model.embed_query("이 대화에서 언급된 이름은 홍길동입니다.")

from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

print(cos_sim(embedded_query_q, embedded_query_a))


from langchain.embeddings import HuggingFaceEmbeddings

model_name = "jhgan/ko-sbert-nli"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
ko = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

sentences = [
    "안녕하세요",
    "제 이름은 홍길동입니다.",
    "이름이 무엇인가요?",
    "랭체인은 유용합니다.",
    "홍길동 아버지의 이름은 홍상직입니다."
    ]

ko_embeddings = ko.embed_documents(sentences)

q = "홍길동은 아버지를 아버지라 부르지 못하였습니다. 홍길동 아버지의 이름은 무엇입니까?"
a = "홍길동의 아버지는 엄했습니다."
ko_query_q = ko.embed_query(q)
ko_query_a = ko.embed_query(a)

print("질문: {} \n".format(q), "-"*100)
print("{} \t\t 문장 유사도: ".format(a), round(cos_sim(ko_query_q, ko_query_a),2))
print("{}\t\t\t 문장 유사도: ".format(sentences[1]), round(cos_sim(ko_query_q, ko_embeddings[1]),2))
print("{}\t\t\t 문장 유사도: ".format(sentences[3]), round(cos_sim(ko_query_q, ko_embeddings[3]),2))
print("{}\t 문장 유사도: ".format(sentences[4]), round(cos_sim(ko_query_q, ko_embeddings[4]),2))
