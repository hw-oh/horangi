# 한국어 벤치마크 개발 가이드

이 문서는 새로운 한국어 벤치마크를 추가하는 방법을 설명합니다.

## 아키텍처

```
benchmarks/
├── config.py      # 벤치마크 설정 정의
├── factory.py     # Task 생성 로직
├── scorers.py     # 커스텀 scorer (필요 시)
├── __init__.py    # 자동 Task 생성 및 노출
└── data/          # 로컬 데이터셋
    └── ko_aime2025.jsonl
```

## 새 벤치마크 추가하기

### Step 1: config.py에 설정 추가

```python
# config.py
BENCHMARKS = {
    # 기존 벤치마크들...
    
    "ko_new_benchmark": {
        # 필수 필드
        "base": "original_benchmark",    # inspect_evals 벤치마크 이름
        "data_type": "weave",            # "weave" 또는 "jsonl"
        "data_source": "weave:///...",   # Weave ref 또는 파일 경로
        "field_mapping": {               # 데이터 필드 → Sample 필드
            "id": "id",
            "input": "question",
            "target": "answer",
            "choices": "options",        # 객관식일 때만
        },
        
        # 선택 필드
        "weave_project": "entity/project",  # Weave 사용 시
        "split": "train",                    # 데이터 분할
        "target_transform": "label_to_letter",  # target 변환
        "system_message": "한국어 시스템 메시지",
        "custom_scorer": "my_scorer",        # 커스텀 scorer 이름
        "metadata": {
            "task_type": "reasoning",
        },
    },
}
```

### Step 2: (필요 시) 커스텀 Scorer 추가

기본 scorer로 처리할 수 없는 경우에만 `scorers.py`에 추가:

```python
# scorers.py
from inspect_ai.scorer import scorer, Scorer, Score, accuracy

@scorer(metrics=[accuracy()])
def my_scorer() -> Scorer:
    async def score(state, target):
        answer = state.output.completion
        # 커스텀 채점 로직
        ...
        return Score(value=..., answer=answer)
    return score
```

### Step 3: eval_tasks.py에 import 추가

```python
# eval_tasks.py
from horangi.benchmarks import (
    ko_hellaswag,
    ko_aime2025,
    ko_new_benchmark,  # 새로 추가
)
```

### Step 4: 테스트

```bash
inspect eval eval_tasks.py@ko_new_benchmark --model openai/gpt-4o -T limit=5
```

---

## 설정 필드 상세

### data_type

| 값 | 설명 |
|----|------|
| `"weave"` | Weave에서 데이터 로드 |
| `"jsonl"` | 로컬 JSONL 파일에서 로드 |

### field_mapping

데이터 필드를 Inspect AI Sample 필드로 매핑:

```python
"field_mapping": {
    "id": "id",                    # 단일 필드
    "id": ["id", "idx"],          # 여러 필드 중 첫 번째 있는 것
    "input": "question",           # 필수
    "target": "answer",            # 필수
    "choices": "options",          # 객관식일 때
}
```

### target_transform

| 값 | 설명 |
|----|------|
| `"identity"` | 변환 없음 (기본값) |
| `"label_to_letter"` | 0→A, 1→B, 2→C, 3→D |
| `"to_string"` | 숫자를 문자열로 변환 |

### base (inspect_evals 벤치마크)

| 값 | Solver | Scorer |
|----|--------|--------|
| `"hellaswag"` | system_message + multiple_choice | choice |
| `"aime2025"` | aime2024_solver | aime_scorer |
| `"gsm8k"` | gsm8k_solver | expression_equivalence |

---

## 예시: KoBBQ 추가

```python
# config.py
"ko_bbq": {
    "base": "bbq",
    "data_type": "weave",
    "data_source": "weave:///wandb-korea/evaluation-job/object/KoBBQ:...",
    "weave_project": "wandb-korea/evaluation-job",
    "field_mapping": {
        "id": "id",
        "input": "context",
        "target": "label",
        "choices": "options",
    },
    "target_transform": "label_to_letter",
    "custom_scorer": "ko_bbq_scorer",  # BBQ는 특수 채점 필요
    "system_message": "주어진 상황을 읽고 질문에 답하세요.",
},
```

```python
# scorers.py
@scorer(metrics=[accuracy()])
def ko_bbq_scorer() -> Scorer:
    async def score(state, target):
        # BBQ bias 측정 로직
        ...
    return score
```

---

## 실행 옵션

```bash
# 기본 실행
inspect eval eval_tasks.py@ko_hellaswag --model openai/gpt-4o

# 옵션
-T limit=10              # 샘플 수 제한
-T shuffle=true          # 데이터 셔플
-T split=validation      # 데이터 분할 (weave)
-T use_korean_prompt=false  # 원본 영어 프롬프트 사용
```

---

## 현재 등록된 벤치마크

| 이름 | Base | 데이터 | 설명 |
|------|------|--------|------|
| `ko_hellaswag` | hellaswag | Weave | 문장 완성 상식 추론 |
| `ko_aime2025` | aime2025 | JSONL | 수학 경시대회 문제 |
