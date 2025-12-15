"""
BFCL - Berkeley Function Calling Leaderboard 벤치마크 (통합 버전)

모델의 tool calling 지원 여부에 따라 자동으로 적절한 solver를 선택합니다.
- Native Tool Calling (기본): tool calling 지원 모델 (OpenAI, Claude, Gemini 등)
- Text-based: tool calling 미지원 모델 (EXAONE, 일부 오픈소스 등)

모델 설정 (configs/models/<model>.yaml):
    benchmarks:
      bfcl:
        use_native_tools: true  # 또는 false

지원 split (150개 샘플):
- simple: 단일 함수 호출 (30개)
- multiple: 여러 함수 중 선택 (30개)
- exec_simple: 실행 가능한 단순 호출 (30개)
- exec_multiple: 실행 가능한 다중 호출 (30개)
- irrelevance: 관련 없는 함수 거부 (30개)

제외:
- parallel*: 병렬 호출
- multi_turn*: 멀티턴 대화
"""

from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    data_type="weave",
    data_source="weave:///wandb-korea/evaluation-job/object/BFCL_Extended:latest",
    field_mapping={
        "id": "id",
        "input": "input",
        # target은 없음 - metadata의 ground_truth 사용
    },
    answer_format="identity",
    # solver는 동적으로 결정됨 (use_native_tools 설정에 따라)
    # 기본값: bfcl_solver (native tool calling)
    solver="bfcl_solver",
    scorer="bfcl_scorer",
    # balanced sampling으로 각 카테고리에서 균등하게 추출
    sampling="balanced",
    sampling_by="category",
    # 메타데이터: 이 벤치마크가 동적 solver 선택을 지원함을 표시
    metadata={
        "supports_dynamic_solver": True,
        "native_solver": "bfcl_solver",
        "text_solver": "bfcl_text_solver",
    },
)

