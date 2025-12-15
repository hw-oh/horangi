"""
KoGSM8K - 한국어 GSM8K 수학 벤치마크

한국어 데이터는 answer가 숫자로만 되어 있어서
원본 gsm8k의 record_to_sample을 사용할 수 없음.
독립 벤치마크로 설정.
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # base 없음 - 독립 벤치마크
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/GSM8K:H1P7QW8yt60sWy6jXXXkBQH8VElTxB3jf9KCFPzqpO4",
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
    },
    system_message="다음 수학 문제를 풀어주세요. 마지막에 풀이 과정과 함께 최종 답을 '\\boxed{{정답}}' 형식으로 명확하게 제시해주세요. 만약 문제가 객관식이라면 객관식의 정답번호를 최종 답으로 제시해주세요.\n최종 답 예시\n- 주관식인 경우: '\\boxed{{7}}'\n- 지문이 1, 2, 3, 4인 객관식인 경우: '\\boxed{{2}}'\n\n문제:",
    answer_format="to_string",  # 숫자 → 문자열
    solver="generate",
    scorer="model_graded_qa",
)
