from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote import logging
from dotenv import load_dotenv


load_dotenv()

# 프로젝트 이름을 입력합니다.
logging.langsmith("CH03-OutputParser")
# 실시간 출력을 위한 import


# 콤마로 구분된 리스트 출력 파서 초기화
output_parser = CommaSeparatedListOutputParser()

# 출력 형식 지침 가져오기
format_instructions = output_parser.get_format_instructions()
# 프롬프트 템플릿 설정
prompt = PromptTemplate(
    # 주제에 대한 다섯 가지를 나열하라는 템플릿
    template="List five {subject}.\n{format_instructions}",
    input_variables=["subject"],  # 입력 변수로 'subject' 사용
    # 부분 변수로 형식 지침 사용
    partial_variables={"format_instructions": format_instructions},
)

# ChatOpenAI 모델 초기화
model = ChatOpenAI(temperature=0)

# 프롬프트, 모델, 출력 파서를 연결하여 체인 생성
chain = prompt | model | output_parser
# "대한민국 관광명소"에 대한 체인 호출 실행
chain.invoke({"subject": "대한민국 관광명소"})
# 스트림을 순회합니다.
for s in chain.stream({"subject": "대한민국 관광명소"}):
    print(s)  # 스트림의 내용을 출력합니다.
