from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# TextLoader를 사용하여 파일을 로드합니다.
loader = TextLoader("./data/appendix-keywords.txt")

# 문서를 로드합니다.
documents = loader.load()

# 문자 기반으로 텍스트를 분할하는 CharacterTextSplitter를 생성합니다. 청크 크기는 300이고 청크 간 중복은 없습니다.
text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)

# 로드된 문서를 분할합니다.
split_docs = text_splitter.split_documents(documents)

# OpenAI 임베딩을 생성합니다.
embeddings = OpenAIEmbeddings()

# 분할된 텍스트와 임베딩을 사용하여 FAISS 벡터 데이터베이스를 생성합니다.
db = FAISS.from_documents(split_docs, embeddings)
# 데이터베이스를 검색기로 사용하기 위해 retriever 변수에 할당
retriever = db.as_retriever(
    # 검색 유형을 "similarity_score_threshold 으로 설정
    search_type="similarity_score_threshold",
    # 임계값 설정
    search_kwargs={"score_threshold": 0.8},
)

# 관련 문서를 검색
for doc in retriever.invoke("Word2Vec 은 무엇인가요?"):
    print(doc.page_content)
    print("=========================================================")
