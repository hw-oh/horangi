# 🐯 Horangi - 한국어 LLM 벤치마크 평가 프레임워크

[Inspect AI](https://inspect.ai-safety-institute.org.uk/)와 [WandB/Weave](https://wandb.ai/site/weave)를 통합하여 한국어 LLM 평가를 수행하는 프레임워크입니다.

## ✨ 특징

- 🇰🇷 **20개+ 한국어 벤치마크** 지원
- 📊 **WandB/Weave 자동 로깅** - 실험 추적 및 결과 비교
- 🚀 **다양한 모델 지원** - OpenAI, Claude, Gemini, DeepSeek, EXAONE 등
- 🔧 **Config 기반** 벤치마크 정의 - 새 벤치마크를 쉽게 추가
- 🛠️ **CLI 지원** - `horangi` 명령어로 간편 실행

## 📦 설치

### uv 사용

[uv](https://docs.astral.sh/uv/)는 빠르고 현대적인 Python 패키지 관리자입니다.

```bash
# uv 설치 (아직 없다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 저장소 클론
git clone https://github.com/your-repo/inspect_horangi.git
cd inspect_horangi

# 의존성 설치 및 가상환경 생성
uv sync

# 개발 의존성 포함 설치
uv sync --all-extras
```

### 환경 변수 설정

```bash
# 필수
export WANDB_API_KEY=your_wandb_api_key

# 모델별 API 키
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export GOOGLE_API_KEY=your_google_api_key

# DeepSeek, Grok 등 (OpenAI 호환)
export OPENAI_BASE_URL=https://api.deepseek.com  # DeepSeek
export OPENAI_BASE_URL=https://api.x.ai/v1       # Grok
```

## 🚀 빠른 시작

### CLI 사용 (권장)

```bash
# 지원 벤치마크 목록 확인
uv run horangi --list

# 벤치마크 실행
uv run horangi kmmlu --model openai/gpt-4o -T limit=10

# 전체 데이터셋
uv run horangi kmmlu --model openai/gpt-4o
```

## 📊 지원 벤치마크

```bash
# 전체 목록 확인
uv run horangi --list
```

### 범용언어성능 (GLP)

| 중구분 | 소구분 | 벤치마크 | 상세 | 구현 |
|--------|--------|----------|------|:----:|
| **기본언어성능** | 구문해석 | `ko_balt_700` | 문장 구조 분석 능력, 문법적 타당성, 문장 성분 간 관계 파악 능력 평가 | ✅ |
| | 의미해석 | `haerae_bench_v1_rc` | 자연어 의미 해석력, 문맥 기반 추론 능력, 의미적 일관성 평가 | ✅ |
| **응용언어성능** | 표현 | `mtbench_ko` | 상황/역할 기반 응답 품질, 글쓰기 능력, 인문학적 표현력 평가 (LLM Judge) | ✅ |
| | ~~번역~~ | ~~korean-parallel-corporal~~ | ~~한영 / 영일 번역~~ | ❌ |
| | 정보검색 | `squad_kor_v1` | 한국어 질의응답 기반 정보검색 능력 및 증거문구 기반 정답 도출 능력 | ✅ |
| **지식/질의응답** | 일반적지식 | `kmmlu`, `haerae_bench_v1_wo_rc` | 상식, 기초학문(STEM), 멀티턴 질의응답 기반의 폭넓은 일반지식 이해도 평가 | ✅ |
| | 전문적지식 | `kmmlu_pro`, `ko_hle` | 의학, 법률, 공학 등 고난도 전문 지식 문제 해결 능력 | ✅ |
| **추론능력** | 상식적 추론 | `ko_hellaswag` | 문장 완성, 다음 문장 예측을 통한 상식 추론 능력 평가 | ✅ |
| | 수학적 추론 | `ko_gsm8k`, `ko_aime2025` | 수학 문제 풀이 능력, 연산/정리/추론 정확도, 복잡한 문제 해결 과정 평가 | ✅ |
| | 논리적 추론 | `mtbench_ko` (reasoning) | 논리적 일관성, 단계별 추론 체계성, 원인-결과 기반 문제 해결능력 측정 | ✅ |
| | 추상적 추론 | `ko_arc_agi` | 시각적/구조적 추론을 포함한 추상적 문제 해결 평가 | ✅ |
| **어플리케이션 개발** | 코딩 | `swebench_verified_official_80` | SWE-bench 기반 실제 GitHub 이슈 해결 능력 평가 | ✅ |
| | 함수호출 | `bfcl_extended`, `bfcl_text` | 함수 호출의 정확성 (단일, 멀티턴, 무관계검출) | ✅ |

### 가치정렬성능 (ALT)

| 중구분 | 소구분 | 벤치마크 | 상세 | 구현 |
|--------|--------|----------|------|:----:|
| **제어성** | 제어성 | `ifeval_ko` | 지시문 수행 능력, 사용자의 명령을 정확하고 일관되게 따르는 능력 평가 | ✅ |
| **윤리/도덕** | 윤리/도덕 | `ko_moral` | 사회 규범 준수, 안전한 언어 생성 능력 평가 | ✅ |
| **유해성방지** | 유해성방지 | `korean_hate_speech` | 혐오발언, 공격적 발화, 위험 카테고리 탐지 및 억제 능력 평가 | ✅ |
| **편향성방지** | 편향성방지 | `kobbq` | 한국어 기반 편향성 평가, 특정 집단/속성에 대한 부적절한 일반화 점검 | ✅ |
| **환각방지** | 환각방지 | `ko_hallulens_*`, `ko_truthful_qa` | 사실성 검증, 근거 기반 답변 생성, 지식 환각 억제 능력 평가 | ✅ |

### 벤치마크 상세

<details>
<summary><b>HalluLens 환각 평가 (5종)</b></summary>

| 벤치마크 | 설명 | 메트릭 |
|----------|------|--------|
| `ko_hallulens_wikiqa` | 짧은 위키 QA | Correct/Hallucination/Refusal |
| `ko_hallulens_longwiki` | 긴 위키 QA | Correct/Hallucination/Refusal |
| `ko_hallulens_generated` | 가상 엔티티 거부 | Refusal Rate |
| `ko_hallulens_mixed` | 혼합 엔티티 거부 | Refusal Rate |
| `ko_hallulens_nonexistent` | 가상 엔티티 통합 | Refusal Rate |

</details>

<details>
<summary><b>MT-Bench 한국어 (8 카테고리)</b></summary>

| 카테고리 | 설명 |
|----------|------|
| `writing` | 글쓰기 능력, 블로그/이메일 등 |
| `roleplay` | 역할극 수행 능력 |
| `reasoning` | 논리적 추론 |
| `math` | 수학 문제 해결 |
| `coding` | 코딩 문제 해결 |
| `extraction` | 정보 추출 |
| `stem` | STEM 지식 |
| `humanities` | 인문학 지식 |

- **80개 질문** (카테고리당 10개)
- **2턴 대화** (Turn 1 → 응답 → Turn 2 → 응답)
- **LLM Judge** 1-10점 평가

</details>

<details>
<summary><b>BFCL Function Calling (9 카테고리)</b></summary>

| 벤치마크 | 모드 | 카테고리 |
|----------|------|----------|
| `bfcl_extended` | Native Tool Calling | simple, multiple, irrelevance, java, javascript |
| `bfcl_text` | Text-based (프롬프트) | live_simple, live_multiple, live_relevance, live_irrelevance |

</details>

<details>
<summary><b>SWE-bench Verified (코딩)</b></summary>

| 벤치마크 | 설명 |
|----------|------|
| `swebench_verified_official_80` | 80개 검증된 GitHub 이슈 해결 |

- **실제 오픈소스 이슈** 기반 패치 생성
- **외부 채점 서버** 사용 (Docker 환경 불필요)
- **Unified Diff 형식** 패치 생성

</details>

## 🔧 옵션

```bash
# 공통 옵션
-T limit=100          # 샘플 수 제한
-T shuffle=true       # 데이터 셔플
-T split=train        # 데이터 분할

# 모델 옵션
--model openai/gpt-4o
--model anthropic/claude-4-5-sonnet
--model google/gemini-3-pro
```

## 📁 프로젝트 구조

```
inspect_horangi/
├── horangi.py              # @task 함수 정의 (진입점)
├── pyproject.toml          # 프로젝트 설정 및 의존성
├── uv.lock                 # 의존성 lock 파일
├── src/
│   ├── benchmarks/         # 벤치마크 설정 파일
│   │   ├── __init__.py     # 벤치마크 등록 및 목록
│   │   ├── ko_hellaswag.py
│   │   ├── kmmlu.py
│   │   └── ...
│   ├── core/               # 핵심 로직
│   │   ├── factory.py      # Task 생성 팩토리
│   │   ├── loaders.py      # 데이터 로딩
│   │   ├── benchmark_config.py  # BenchmarkConfig 데이터클래스
│   │   └── answer_format.py
│   ├── scorers/            # 커스텀 Scorer
│   │   ├── bfcl_scorer.py
│   │   ├── kobbq_scorer.py
│   │   ├── hallulens_qa_scorer.py
│   │   ├── swebench_server_scorer.py
│   │   └── ...
│   ├── solvers/            # 커스텀 Solver
│   │   ├── bfcl_solver.py
│   │   └── swebench_patch_solver.py
│   └── cli/                # CLI 엔트리포인트
│       └── __init__.py
└── create_benchmark/       # 데이터셋 생성 스크립트
```

## 🔌 모델 지원

### Native 지원 (추가 설정 불필요)

| Provider | 모델 예시 |
|----------|-----------|
| OpenAI | `openai/gpt-4o`, `openai/gpt-4o-mini` |
| Anthropic | `anthropic/claude-3-5-sonnet-20241022` |
| Google | `google/gemini-1.5-pro` |
| Mistral | `mistral/mistral-large-latest` |
| Together | `together/meta-llama/Llama-3-70b-chat-hf` |

### OpenAI 호환 API

```bash
# DeepSeek
export OPENAI_BASE_URL=https://api.deepseek.com
uv run horangi kmmlu --model openai/deepseek-chat

# Grok (xAI)
export OPENAI_BASE_URL=https://api.x.ai/v1
uv run horangi kmmlu --model openai/grok-beta
```

### 로컬/자체 모델

```bash
# vLLM
uv run horangi kmmlu --model vllm/LGAI-EXAONE/EXAONE-3.5-32B-Instruct

# Ollama
uv run horangi kmmlu --model ollama/llama3.1:70b
```

## 📈 결과 확인

### Weave Evaluation

Weave UI에서 상세 결과 확인:
- 샘플별 점수 및 응답
- 모델 간 비교
- 집계 메트릭 (Scores 섹션)

## 🔧 inspect-wandb Fork

이 프로젝트는 Weave 통합을 위해 fork된 [inspect-wandb](https://github.com/hw-oh/inspect_wandb)를 사용합니다:

- Weave UI의 Scores 섹션에 집계 메트릭 표시
- CORRECT/INCORRECT 값을 boolean으로 변환하여 수치 집계 지원

## 📚 참고 자료

- [Inspect AI Documentation](https://inspect.ai-safety-institute.org.uk/)
- [inspect-wandb (fork)](https://github.com/hw-oh/inspect_wandb)
- [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)
- [WandB Weave](https://wandb.ai/site/weave)

## 📄 라이선스

MIT License
