# 🔍 Weave 통합 가이드

이 문서는 Horangi 프로젝트에서 [WandB Weave](https://wandb.ai/site/weave)를 활용하는 방법을 설명합니다.

---

## 📖 Weave란?

**Weave**는 Weights & Biases에서 제공하는 LLM 애플리케이션 관찰성(Observability) 및 평가(Evaluation) 도구입니다.

### 주요 기능

| 기능 | 설명 |
|------|------|
| **Traces** | LLM 호출의 입력/출력, 토큰 사용량, 지연시간 등을 자동으로 추적 |
| **Evaluations** | 벤치마크 평가 결과를 구조화된 형태로 저장하고 비교 |
| **Datasets** | 평가용 데이터셋을 버전 관리하며 저장 |
| **Leaderboard** | 여러 모델의 평가 결과를 한눈에 비교하는 리더보드 생성 |


---

## 🔗 Weave 프로젝트 구조

Horangi 평가를 실행하면 다음과 같은 Weave 객체들이 생성됩니다:

```
Weave Project (<entity>/<project>)
├── Objects/
│   ├── Datasets (벤치마크 데이터셋)
│   │   ├── KoHellaSwag_mini
│   │   ├── KMMLU_mini
│   │   └── ...
│   └── Leaderboards
│       └── Korean-LLM-Leaderboard
├── Evaluations/
│   ├── ko_hellaswag-evaluation
│   ├── kmmlu-evaluation
│   └── ...
└── Traces/
    └── (각 평가의 상세 실행 기록)
```

---

## 📊 Evaluations (평가 결과)

> 📚 **공식 문서**: [Evaluations Overview](https://docs.wandb.ai/weave/guides/core-types/evaluations)

### 개요

벤치마크를 실행하면 각 평가가 **Evaluation** 객체로 저장됩니다. Evaluation은 다음 정보를 포함합니다:

- **Model**: 평가된 모델 이름
- **Dataset**: 사용된 데이터셋
- **Scores**: 각 scorer별 집계 점수
- **Samples**: 개별 샘플의 입력/출력/점수

### 평가 실행 후 결과 확인

```bash
uv run horangi kmmlu --config gpt-4o
```

실행이 완료되면 터미널에 Weave URL이 출력됩니다:

```
🔗 Weave Eval 예시: https://wandb.ai/horangi/horangi4/r/call/019b2a28-...
```

![Evaluation 결과 화면 스크린샷](./assets/evals.png)

### Evaluation UI 구성

| 탭 | 설명 |
|----|------|
| **Evaluation** | 평가 결과 요약 및 샘플별 결과 테이블 |
| **Call** | 평가 실행의 호출 정보 |
| **Feedback** | 사용자 피드백 (있는 경우) |
| **Summary** | 평가 메타데이터 요약 |
| **Use** | 코드에서 참조하는 방법 |

### 화면 구성 요소

**Definition**: 평가의 기본 정보
- `kmmlu-evaluation:v23` - Evaluation 객체 버전
- `openai-gpt_4o:v0` - 평가된 모델
- `kmmlu:v0` - 사용된 데이터셋

**Scores**: 집계된 점수
- `choice` - 정답률 (예: 64 of 100 → 64.0%)
- `total_time` - 평균 응답 시간
- `total_tokens` - 평균 토큰 사용량

**Results**: 샘플별 결과 테이블
- `input` - 질문 내용
- `Output` - 모델의 응답 (예: ANSWER: D)
- `choice` - 정답 여부 (✓/✗)
- `total_time` - 응답 시간
- `total_tokens` - 토큰 수

---

## 🔎 Traces (실행 추적)

> 📚 **공식 문서**: [Tracing Quickstart](https://docs.wandb.ai/weave/guides/tracking/tracing)

### 개요

**Traces**는 평가 과정에서 발생하는 모든 LLM 호출을 샘플 단위로 기록합니다.

각 Trace에는 다음 정보가 포함됩니다:
- **입력/출력**: 모델에 전달된 프롬프트와 생성된 응답
- **점수**: Scorer가 부여한 정답 여부 및 세부 점수
- **성능 지표**: 응답 시간, 토큰 사용량

이를 통해 **오답 분석**, **응답 시간 병목 파악**, **토큰 비용 최적화** 등을 수행할 수 있습니다.

Evaluations 화면에서 `View traces` 버튼을 눌러 해당 평가에 기록된 트레이스를 확인할 수 있습니다.

### Traces 테이블 컬럼

| 컬럼 | 설명 |
|------|------|
| **Trace** | Trace 이름 (`Evaluation.predict_and_score` + 해시) |
| **Feedback** | 사용자 피드백 (있는 경우) |
| **Status** | 실행 상태 (✓ 성공) |
| **...input** | 입력 질문 내용 |
| **model** | 사용된 모델 (예: `openai-gpt_4o...`) |
| **self** | 연결된 Evaluation (예: `kmmlu-evalua...`) |
| **output** | 모델 응답 (예: `ANSWER: A`) |
| **...choice** | 정답 여부 (✓/✗) |
| **...total_time** | 응답 시간 (초) |
| **...total_tokens** | 사용된 토큰 수 |


![Traces 목록 화면 스크린샷](./assets/traces.png)

### Trace 상세 보기

개별 Trace를 클릭하면 전체 호출 체인을 확인할 수 있습니다:

![Trace 상세 화면 스크린샷](./assets/trace_detail.png)

---

## 🏆 Leaderboard (리더보드)

> 📚 **공식 문서**: [Leaderboard Quickstart](https://docs.wandb.ai/weave/cookbooks/leaderboard_quickstart)

### 개요

**Leaderboard**는 여러 모델의 평가 결과를 한눈에 비교할 수 있는 테이블입니다.

![Weave Leaderboard](./assets/leaderboard.png)

---

## ⚙️ Weave 설정

### 환경 변수

Weave 연동을 위해 다음 환경 변수가 필요합니다:

```bash
# .env 파일
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=horangi          # 또는 본인의 entity
WANDB_PROJECT=horangi4        # 또는 본인의 project
```

---

## 🔗 유용한 링크

### 공식 문서
- [Weave Documentation](https://docs.wandb.ai/weave)
- [Weave Cookbooks](https://docs.wandb.ai/weave/cookbooks) - 다양한 활용 예제
- [Evaluations Guide](https://docs.wandb.ai/weave/guides/core-types/evaluations)
- [Tracing Guide](https://docs.wandb.ai/weave/guides/tracking/tracing)
- [Leaderboard Quickstart](https://docs.wandb.ai/weave/cookbooks/leaderboard_quickstart)
- [호랑이 리더보드](https://horangi.ai)
