# 벤치마크 추가 가이드

> 새 벤치마크 추가 방법을 설명합니다.
> 설치, 사용법, 모델 설정은 [루트 README](../README.md)를 참고하세요.

---

## 🎯 새 벤치마크 추가

### Step 1: Config 파일 생성

```python
# src/benchmarks/my_benchmark.py
from core.benchmark_config import BenchmarkConfig

CONFIG = BenchmarkConfig(
    # 데이터 소스 (필수)
    data_type="weave",  # "weave" 또는 "jsonl"
    data_source="weave:///entity/project/object/DatasetName:latest",
    
    # 필드 매핑
    field_mapping={
        "id": "id",
        "input": "question",
        "target": "answer",
        "choices": "options",  # MCQA만
    },
    
    # 평가 설정
    answer_format="index_0",
    solver="multiple_choice",
    scorer="choice",
    system_message="시스템 프롬프트",
)
```

### Step 2: 등록

```python
# src/benchmarks/__init__.py
from benchmarks.my_benchmark import CONFIG as my_benchmark

BENCHMARKS = {
    ...
    "my_benchmark": my_benchmark,
}

BENCHMARK_DESCRIPTIONS = {
    ...
    "my_benchmark": "벤치마크 설명",
}
```

### Step 3: Task 함수 추가

```python
# horangi.py (루트)
@task
def my_benchmark(shuffle: bool = False, limit: int | None = None) -> Task:
    """My Benchmark"""
    return create_benchmark(name="my_benchmark", shuffle=shuffle, limit=limit)
```

### Step 4: 테스트

```bash
uv run horangi my_benchmark --model openai/gpt-4o -T limit=5
```

---

## 📋 BenchmarkConfig 필드 참조

### 필수 필드

| 필드 | 설명 |
|------|------|
| `data_type` | `"weave"` 또는 `"jsonl"` |
| `data_source` | Weave URI 또는 JSONL 파일명 (`src/data/` 기준) |

### 주요 선택 필드

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `field_mapping` | `{}` | 데이터셋 → Sample 필드 매핑 |
| `solver` | `"multiple_choice"` | Solver |
| `scorer` | `"choice"` | Scorer |
| `answer_format` | `"index_0"` | 정답 변환 방식 |
| `system_message` | `None` | 시스템 프롬프트 |

### `answer_format` 옵션

| 값 | 설명 | 예시 |
|----|------|------|
| `identity` | 변환 없음 | `"정답"` → `"정답"` |
| `index_0` | 0-indexed → A/B/C | `0` → `"A"` |
| `index_1` | 1-indexed → A/B/C | `1` → `"A"` |
| `text` | 텍스트 → 인덱스 | `"사과"` → `"A"` |
| `letter` | 그대로 유지 | `"A"` → `"A"` |
| `to_string` | 문자열로 변환 | `123` → `"123"` |
| `boolean` | True/False → A/B | `True` → `"A"` |

### Solver / Scorer 옵션

| Solver | 용도 |
|--------|------|
| `multiple_choice` | MCQA |
| `generate` | 자유 생성 |
| `bfcl_solver` | Tool calling (Native) |
| `bfcl_text_solver` | Tool calling (Text-based) |
| `mtbench_solver` | MT-Bench 멀티턴 대화 |
| `swebench_patch_solver` | SWE-bench |

| Scorer | 용도 |
|--------|------|
| `choice` | MCQA 정확도 |
| `match` | 정확 일치 |
| `match_numeric` | 숫자 일치 |
| `model_graded_qa` | LLM 채점 |
| `hle_grader` | HLE 전용 채점 |
| `grid_match` | 그리드 일치 (ARC-AGI) |
| `macro_f1` | Macro F1 |
| `kobbq_scorer` | KoBBQ 편향성 |
| `hallulens_qa_scorer` | HalluLens QA |
| `refusal_scorer` | HalluLens 거부 응답 평가 |
| `bfcl_scorer` | BFCL 함수호출 |
| `mtbench_scorer` | MT-Bench 평가 |
| `swebench_server_scorer` | SWE-bench 서버 채점 |

---

## 🔧 커스텀 Scorer 추가

### Step 1: Scorer 파일 생성

```python
# src/scorers/my_scorer.py
from inspect_ai.scorer import Score, Scorer, Target, scorer, accuracy, CORRECT, INCORRECT
from inspect_ai.solver import TaskState

@scorer(metrics=[accuracy()])
def my_scorer() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        response = state.output.completion
        expected = target.text
        is_correct = response.strip() == expected.strip()
        
        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=response[:100],
        )
    return score
```

### Step 2: 등록

```python
# src/scorers/__init__.py
from scorers.my_scorer import my_scorer

__all__ = [..., "my_scorer"]
```

---

## 📝 체크리스트

새 벤치마크 추가 시:

- [ ] `src/benchmarks/`에 config 파일 생성
- [ ] `src/benchmarks/__init__.py`에 등록
- [ ] `horangi.py`에 `@task` 함수 추가
- [ ] 테스트 실행

---

## 🔗 참고

- [Inspect AI Docs](https://inspect.ai-safety-institute.org.uk/)
- [inspect_evals GitHub](https://github.com/UKGovernmentBEIS/inspect_evals)

