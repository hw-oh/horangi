# 한국어 벤치마크 개발 가이드

이 문서는 새로운 한국어 벤치마크를 추가하는 방법을 설명합니다.

## 벤치마크 구현 패턴

벤치마크를 구현할 때 두 가지 패턴 중 하나를 선택합니다:

| 패턴 | 설명 | 사용 시점 |
|------|------|----------|
| **상속 패턴** | 기존 `inspect_evals` 벤치마크 재사용 | 원본 벤치마크가 있고, 데이터만 교체할 때 |
| **독립 패턴** | 처음부터 새로 구현 | 원본이 없거나 구조가 다를 때 |

---

## 패턴 1: 상속 패턴 (Inherited)

기존 `inspect_evals`의 벤치마크를 기반으로 데이터 소스만 교체합니다.

### 언제 사용하나요?

- ✅ 원본 영어 벤치마크가 `inspect_evals`에 존재할 때
- ✅ 원본과 동일한 평가 로직을 유지하고 싶을 때
- ✅ 데이터 필드 구조가 원본과 동일할 때

### 장점

- 원본 벤치마크와 동일한 평가 로직 보장
- 코드 중복 최소화
- 원본 업데이트 시 자동 반영

### 단점

- `inspect_evals` 의존성 필요
- 원본 구조 변경 시 호환성 문제 가능

### 템플릿

```python
"""
Ko{BenchmarkName} (상속 패턴)

inspect_evals.{benchmark_name}를 기반으로 데이터 소스만 Weave로 교체합니다.
"""

from typing import Any, Literal

import weave
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from inspect_ai.scorer import choice  # 또는 match, model_graded_qa 등
from inspect_ai.solver import multiple_choice, system_message  # 필요한 solver

# ============================================================================
# 원본 벤치마크에서 재사용
# ============================================================================
from inspect_evals.{benchmark_name}.{benchmark_name} import (
    SYSTEM_MESSAGE as ORIGINAL_SYSTEM_MESSAGE,
    # record_to_sample,  # 구조가 같으면 직접 사용 가능
)

# ============================================================================
# 한국어 벤치마크 설정
# ============================================================================
WEAVE_REF = "weave:///{entity}/{project}/object/{DatasetName}:{version}"

SYSTEM_MESSAGE_KO = """
(원본 시스템 메시지의 한국어 번역)
"""


def record_to_sample(record: dict[str, Any]) -> Sample:
    """
    원본 벤치마크의 record_to_sample과 동일한 로직 적용
    
    데이터 필드 매핑:
    - {원본필드} → {Sample필드}
    """
    # 원본 로직 참고하여 구현
    return Sample(
        id=str(record.get("id", "")),
        input=record["input_field"],
        target=record["target_field"],
        choices=record.get("choices", None),  # 객관식인 경우
        metadata={...},
    )


def load_dataset_from_weave(
    split: Literal["train", "validation", "test"] | None = None,
    shuffle: bool = False,
    limit: int | None = None,
) -> Dataset:
    """Weave에서 데이터셋 로드"""
    weave.init("{entity}/{project}")
    data = weave.ref(WEAVE_REF).get()
    
    rows = data.rows if hasattr(data, "rows") else list(data)
    
    if split:
        rows = [r for r in rows if r.get("split") == split]
    
    samples = [record_to_sample(r) for r in rows]
    
    if limit:
        samples = samples[:limit]
    
    return MemoryDataset(samples=samples, shuffled=shuffle)


@task
def ko_{benchmark_name}_inherited(
    shuffle: bool = False,
    split: Literal["train", "validation", "test"] | None = None,
    limit: int | None = None,
    use_korean_prompt: bool = True,
) -> Task:
    """
    Ko{BenchmarkName} (상속 패턴)
    
    원본 inspect_evals.{benchmark_name}와 동일한 구조 사용
    """
    dataset = load_dataset_from_weave(split=split, shuffle=shuffle, limit=limit)
    
    sys_msg = SYSTEM_MESSAGE_KO if use_korean_prompt else ORIGINAL_SYSTEM_MESSAGE
    
    # 원본과 동일한 Task 구조
    return Task(
        dataset=dataset,
        solver=[system_message(sys_msg), multiple_choice()],  # 원본과 동일
        scorer=choice(),  # 원본과 동일
        name="ko_{benchmark_name}_inherited",
        metadata={
            "benchmark": "ko_{benchmark_name}",
            "pattern": "inherited",
            "base": "inspect_evals.{benchmark_name}",
            "language": "ko",
            "split": split,
        },
    )
```

### 예시: KoHellaSwag

```python
# 원본 참조
from inspect_evals.hellaswag.hellaswag import SYSTEM_MESSAGE

# 원본과 동일한 구조
return Task(
    dataset=dataset,
    solver=[system_message(sys_msg), multiple_choice()],
    scorer=choice(),
)
```

---

## 패턴 2: 독립 패턴 (Standalone)

`inspect_evals` 없이 처음부터 새로 구현합니다.

### 언제 사용하나요?

- ✅ 원본 벤치마크가 `inspect_evals`에 없을 때
- ✅ 완전히 새로운 한국어 벤치마크를 만들 때
- ✅ 원본과 구조가 많이 다를 때
- ✅ 외부 의존성을 최소화하고 싶을 때

### 장점

- 외부 의존성 없음
- 완전한 커스터마이징 가능
- 원본 변경에 영향받지 않음

### 단점

- 코드 중복 발생 가능
- 원본과 동일한 평가 로직 보장 어려움

### 템플릿

```python
"""
Ko{BenchmarkName} (독립 패턴)

inspect_evals 없이 독립적으로 구현한 한국어 벤치마크입니다.
"""

from typing import Any, Literal

import weave
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, Sample, MemoryDataset
from inspect_ai.scorer import choice, match, model_graded_qa  # 필요한 것 선택
from inspect_ai.solver import generate, multiple_choice, system_message

# ============================================================================
# 벤치마크 설정
# ============================================================================
WEAVE_REF = "weave:///{entity}/{project}/object/{DatasetName}:{version}"

SYSTEM_MESSAGE = """
(한국어 시스템 메시지)
"""


def record_to_sample(record: dict[str, Any]) -> Sample:
    """
    데이터 레코드를 Sample로 변환
    
    데이터 구조:
    - {필드명}: {설명}
    """
    return Sample(
        id=str(record.get("id", "")),
        input=record["input_field"],
        target=record["target_field"],
        choices=record.get("choices", None),
        metadata={...},
    )


def load_dataset_from_weave(
    split: Literal["train", "validation", "test"] | None = None,
    shuffle: bool = False,
    limit: int | None = None,
) -> Dataset:
    """Weave에서 데이터셋 로드"""
    weave.init("{entity}/{project}")
    data = weave.ref(WEAVE_REF).get()
    
    rows = data.rows if hasattr(data, "rows") else list(data)
    
    if split:
        rows = [r for r in rows if r.get("split") == split]
    
    samples = [record_to_sample(r) for r in rows]
    
    if limit:
        samples = samples[:limit]
    
    return MemoryDataset(samples=samples, shuffled=shuffle)


@task
def ko_{benchmark_name}_standalone(
    shuffle: bool = False,
    split: Literal["train", "validation", "test"] | None = None,
    limit: int | None = None,
) -> Task:
    """
    Ko{BenchmarkName} (독립 패턴)
    
    {벤치마크 설명}
    """
    dataset = load_dataset_from_weave(split=split, shuffle=shuffle, limit=limit)
    
    return Task(
        dataset=dataset,
        solver=[
            system_message(SYSTEM_MESSAGE),
            # 평가 유형에 따라 선택:
            # multiple_choice(),  # 객관식
            # generate(),         # 주관식
        ],
        scorer=choice(),  # 또는 match(), model_graded_qa()
        name="ko_{benchmark_name}_standalone",
        metadata={
            "benchmark": "ko_{benchmark_name}",
            "pattern": "standalone",
            "language": "ko",
            "split": split,
        },
    )
```

---

## 진입점 파일 만들기

두 패턴을 모두 제공할 경우, 진입점 파일을 만들어 자동 선택하도록 합니다.

```python
# ko_{benchmark_name}.py

"""
Ko{BenchmarkName}: 한국어 {BenchmarkName} 벤치마크

두 가지 구현 패턴 제공:
1. ko_{benchmark_name}_inherited: inspect_evals 재사용
2. ko_{benchmark_name}_standalone: 독립 구현
"""

# 상속 패턴 (inspect_evals 필요)
try:
    from horangi.benchmarks.ko_{benchmark_name}_inherited import (
        ko_{benchmark_name}_inherited,
        SYSTEM_MESSAGE_KO,
    )
    HAS_INSPECT_EVALS = True
except ImportError:
    HAS_INSPECT_EVALS = False
    ko_{benchmark_name}_inherited = None

# 독립 패턴 (항상 사용 가능)
from horangi.benchmarks.ko_{benchmark_name}_standalone import (
    ko_{benchmark_name}_standalone,
    SYSTEM_MESSAGE,
)

# 기본 선택: 상속 패턴 우선
if HAS_INSPECT_EVALS:
    ko_{benchmark_name} = ko_{benchmark_name}_inherited
else:
    ko_{benchmark_name} = ko_{benchmark_name}_standalone

__all__ = [
    "ko_{benchmark_name}",
    "ko_{benchmark_name}_inherited",
    "ko_{benchmark_name}_standalone",
    "HAS_INSPECT_EVALS",
]
```

---

## eval_tasks.py에 등록

```python
# eval_tasks.py 하단에 추가

from horangi.benchmarks.ko_{benchmark_name} import (
    ko_{benchmark_name},
)
from horangi.benchmarks.ko_{benchmark_name}_inherited import ko_{benchmark_name}_inherited
from horangi.benchmarks.ko_{benchmark_name}_standalone import ko_{benchmark_name}_standalone
```

---

## 체크리스트

새 벤치마크 추가 시 확인사항:

### 공통
- [ ] Weave 데이터셋 ref 확인
- [ ] 데이터 필드 구조 파악
- [ ] `record_to_sample` 함수 구현
- [ ] `load_dataset_from_weave` 함수 구현
- [ ] `@task` 데코레이터 적용
- [ ] metadata에 `benchmark`, `pattern`, `language` 포함

### 상속 패턴
- [ ] `inspect_evals`에서 원본 벤치마크 확인
- [ ] 원본 SYSTEM_MESSAGE 임포트
- [ ] 원본과 동일한 solver/scorer 사용
- [ ] metadata에 `base` 필드 추가

### 독립 패턴
- [ ] SYSTEM_MESSAGE 직접 작성
- [ ] 적절한 solver/scorer 선택
- [ ] 필요한 커스터마이징 적용

### 등록
- [ ] `benchmarks/__init__.py`에 추가
- [ ] `eval_tasks.py`에 추가
- [ ] 테스트 실행 확인

---

## 평가 유형별 Solver/Scorer 조합

| 평가 유형 | Solver | Scorer |
|----------|--------|--------|
| 객관식 (4지선다) | `multiple_choice()` | `choice()` |
| 주관식 (단답형) | `generate()` | `match()` |
| 주관식 (서술형) | `generate()` | `model_graded_qa()` |
| 추론 (CoT) | `chain_of_thought()`, `generate()` | `match()` |

---

## 실행 예시

```bash
# 상속 패턴
inspect eval eval_tasks.py@ko_hellaswag_inherited --model openai/gpt-4o -T limit=5

# 독립 패턴  
inspect eval eval_tasks.py@ko_hellaswag_standalone --model openai/gpt-4o -T limit=5

# 기본 (자동 선택)
inspect eval eval_tasks.py@ko_hellaswag --model openai/gpt-4o -T limit=5
```

