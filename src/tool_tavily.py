# API 키를 환경변수로 관리하기 위한 설정 파일
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_teddynote import logging
from dotenv import load_dotenv

# API 키 정보 로드
load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH15-Tools")

# 도구 생성
tool = TavilySearchResults(
    max_results=6,
    include_answer=True,
    include_raw_content=True,
    # include_images=True,
    # search_depth="advanced", # or "basic"
    include_domains=["github.io", "wikidocs.net"],
    # exclude_domains = []
)
# 도구 실행
result = tool.invoke({"query": "LangChain Tools 에 대해서 알려주세요"})
print(result)
