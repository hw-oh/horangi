# 🐯 Horangi - 한국어 LLM 벤치마크 평가 프레임워크

[Inspect AI](https://inspect.ai-safety-institute.org.uk/)와 [WandB/Weave](https://wandb.ai/site/weave)를 통합하여 한국어 LLM 평가를 수행하는 프레임워크입니다.

## ✨ 특징

- 🇰🇷 **20여개 한국어 벤치마크** 지원
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
| **기본언어성능** | 구문해석 | `ko_balt_700 (syntac)` | 문장 구조 분석 능력, 문법적 타당성, 문장 성분 간 관계 파악 능력 평가 | ✅ |
| | 의미해석 | `haerae_bench_v1 (rc)`, `ko_balt_700 (semantic)` | 자연어 의미 해석력, 문맥 기반 추론 능력, 의미적 일관성 평가 | ✅ |
| **응용언어성능** | 표현 | `mtbench` | 상황/역할 기반 응답 품질, 글쓰기 능력, 인문학적 표현력 평가 (LLM Judge) | ✅ |
| | ~~번역~~ | ~~korean-parallel-corporal~~ | ~~한영 / 영일 번역~~ | ❌ |
| | 정보검색 | `squad_kor_v1` | 한국어 질의응답 기반 정보검색 능력 및 증거문구 기반 정답 도출 능력 | ✅ |
| **지식/질의응답** | 일반적지식 | `kmmlu`, `haerae_bench_v1 (\wo rc)` | 상식, 기초학문(STEM), 멀티턴 질의응답 기반의 폭넓은 일반지식 이해도 평가 | ✅ |
| | 전문적지식 | `kmmlu_pro`, `hle` | 의학, 법률, 공학 등 고난도 전문 지식 문제 해결 능력 | ✅ |
| **추론능력** | 상식적 추론 | `hellaswag` | 문장 완성, 다음 문장 예측을 통한 상식 추론 능력 평가 | ✅ |
| | 수학적 추론 | `gsm8k`, `aime2025` | 수학 문제 풀이 능력, 연산/정리/추론 정확도, 복잡한 문제 해결 과정 평가 | ✅ |
| | 논리적 추론 | `mtbench (reasoning)` | 논리적 일관성, 단계별 추론 체계성, 원인-결과 기반 문제 해결능력 측정 | ✅ |
| | 추상적 추론 | `arc_agi` | 시각적/구조적 추론을 포함한 추상적 문제 해결 평가 | ✅ |
| **어플리케이션 개발** | 코딩 | `swebench_verified_official_80`, `mtbench (coding)` | SWE-bench 기반 실제 GitHub 이슈 해결 능력 평가 | ✅ |
| | 함수호출 | `bfcl` | 함수 호출의 정확성 (단일, 멀티턴, 무관계검출) | ✅ |

### 가치정렬성능 (ALT)

| 중구분 | 소구분 | 벤치마크 | 상세 | 구현 |
|--------|--------|----------|------|:----:|
| **제어성** | 제어성 | `ifeval_ko` | 지시문 수행 능력, 사용자의 명령을 정확하고 일관되게 따르는 능력 평가 | ✅ |
| **윤리/도덕** | 윤리/도덕 | `moral` | 사회 규범 준수, 안전한 언어 생성 능력 평가 | ✅ |
| **유해성방지** | 유해성방지 | `korean_hate_speech` | 혐오발언, 공격적 발화, 위험 카테고리 탐지 및 억제 능력 평가 | ✅ |
| **편향성방지** | 편향성방지 | `kobbq` | 한국어 기반 편향성 평가, 특정 집단/속성에 대한 부적절한 일반화 점검 | ✅ |
| **환각방지** | 환각방지 | `hallulens`, `truthful_qa` | 사실성 검증, 근거 기반 답변 생성, 지식 환각 억제 능력 평가 | ✅ |

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

## 🗜️ 미니 벤치마크 데이터셋

수십 개 모델을 빠르게 평가하기 위한 **미니 버전 데이터셋**을 생성할 수 있습니다.

### 생성 기준

| 기준 | 설명 |
|------|------|
| **기본 샘플 수** | 100개 (원본이 100개 미만이면 전체 사용) |
| **Stratified Sampling** | 카테고리가 있는 데이터셋은 원본 분포 비율 유지 |
| **랜덤 시드** | 42 (재현성 보장) |

### 데이터셋별 샘플 수

| 벤치마크 | 원본 | 미니 | 비고 |
|----------|-----:|-----:|------|
| **ko_hellaswag** | 39,905 | 100 | label 4개 분포 유지 |
| **ko_aime2025** | 30 | 30 | 전체 사용 |
| **ifeval_ko** | 342 | 100 | |
| **haerae_bench_v1** | 1,538 | **200** | rc 100개 + wo_rc 100개 (통합) |
| **ko_balt_700** | 515 | **200** | syntax 100개 + semantic 100개 (통합) |
| **kmmlu** | 35,030 | 100 | category 45개 분포 유지 |
| **kmmlu_pro** | 2,822 | 100 | |
| **squad_kor_v1** | 5,774 | 100 | |
| **ko_truthful_qa** | 817 | 100 | |
| **ko_moral** | 45,215 | 100 | |
| **ko_arc_agi** | 400 | 100 | |
| **ko_gsm8k** | 1,319 | 100 | |
| **korean_hate_speech** | 8,367 | 100 | |
| **kobbq** | 81,128 | 100 | category 10개 분포 유지 |
| **ko_hle** | 2,158 | 100 | category 8개 분포 유지 |
| **ko_hallulens_wikiqa** | 1,433 | 100 | |
| **ko_hallulens_longwiki** | 250 | 100 | |
| **ko_hallulens_nonexistent** | 9,950 | 100 | category 2개 분포 유지 |
| **bfcl** | 258 | 258 | 전체 사용 (category 9개) |
| **mtbench_ko** | 80 | 80 | 전체 사용 (category 8개) |
| **swebench_verified_official_80** | 80 | 80 | 전체 사용 |
| **총합** | **237,411** | **~2,348** | 약 99% 압축 |

### 미니 데이터셋 생성

```bash
# 생성 스크립트 실행
uv run python create_benchmark/create_mini_benchmarks.py

# 출력 디렉토리 지정
uv run python create_benchmark/create_mini_benchmarks.py --output-dir src/data/mini

# 다른 시드 사용
uv run python create_benchmark/create_mini_benchmarks.py --seed 123
```

### 통합 데이터셋 설명

일부 벤치마크는 여러 소스를 합쳐서 하나의 미니 데이터셋으로 생성됩니다:

- **`haerae_bench_v1_mini.jsonl`**: 
  - `haerae_bench_v1_rc` (독해 포함): 100개
  - `haerae_bench_v1_wo_rc` (독해 제외, 5개 카테고리 분포 유지): 100개
  - 총 200개

- **`ko_balt_700_mini.jsonl`**:
  - `ko_balt_700_syntax` (통사론): 100개
  - `ko_balt_700_semantic` (의미론): 100개
  - 총 200개

각 샘플에는 `_source` 필드가 추가되어 원본 데이터셋을 추적할 수 있습니다.

### Weave 미니 데이터셋 참조

미니 데이터셋은 `horangi/horangi4` 프로젝트에 업로드되어 있습니다:

| 데이터셋 | Weave Ref |
|----------|-----------|
| KoHellaSwag_mini | `weave:///horangi/horangi4/object/KoHellaSwag_mini:w5y3uB67dxszTK1uXakGqD2IYKZSrsW1AYQcPH9hIE8` |
| KoAIME2025_mini | `weave:///horangi/horangi4/object/KoAIME2025_mini:ODxXSY7bvgJkZm3bio3ylFSuv3LWzET6aq4SlzkZgUA` |
| IFEval_Ko_mini | `weave:///horangi/horangi4/object/IFEval_Ko_mini:qzHRd8tmmARVui2M4dj4P363Ha8L28XQlvvcrUlrHCM` |
| KMMLU_mini | `weave:///horangi/horangi4/object/KMMLU_mini:BKMMNPwQlldJ6rjGxCPJxEX2thu3XVsEfiYQdf2BHTA` |
| KMMLU_Pro_mini | `weave:///horangi/horangi4/object/KMMLU_Pro_mini:Qbju8ttQj6C4HwI6N2UG7bqB1OnHTZ21IqluhZuiMsM` |
| SQuAD_Kor_v1_mini | `weave:///horangi/horangi4/object/SQuAD_Kor_v1_mini:DXbPOb1F6e8rnKDYJXOhgc5L16ZnaKXrx2EynK4vj6o` |
| KoTruthfulQA_mini | `weave:///horangi/horangi4/object/KoTruthfulQA_mini:aXWwop2uqxplEhdvz576gyfUO4NSkrGNko7hguxueic` |
| KoMoral_mini | `weave:///horangi/horangi4/object/KoMoral_mini:dleEC4Y9ibeC4YAScIEji2CFBX0hXloQX3dvuUubXBo` |
| KoARC_AGI_mini | `weave:///horangi/horangi4/object/KoARC_AGI_mini:HSzsUWJnTXMYwOtS8A6wyfHM1DqsoTugtpBOwmvBuoA` |
| KoGSM8K_mini | `weave:///horangi/horangi4/object/KoGSM8K_mini:xM4iBSffZkeb89tGfn80GDvyV8AplUIww1AiT8E4gp8` |
| KoreanHateSpeech_mini | `weave:///horangi/horangi4/object/KoreanHateSpeech_mini:DBtUl95dG2Xg9qQR49Y250p9oshCMKdkjXdxhvXmLIc` |
| KoBBQ_mini | `weave:///horangi/horangi4/object/KoBBQ_mini:p12gIldwSX2XweDFuDyBJkq09b4X5crbw8tcx73nxR8` |
| KoHLE_mini | `weave:///horangi/horangi4/object/KoHLE_mini:UrNXEnhaUHDoqButTAy204OEEevet6Pa1iSRYfnnnPY` |
| KoHalluLens_WikiQA_mini | `weave:///horangi/horangi4/object/KoHalluLens_WikiQA_mini:rU9poRP5fcXtp7mZsuRYYDNKPK51OkMRJTuXjyXP9WI` |
| KoHalluLens_LongWiki_mini | `weave:///horangi/horangi4/object/KoHalluLens_LongWiki_mini:VktVotlYffXkFz0VT5sKgXrEmItplwFb3R97zb6syEA` |
| KoHalluLens_NonExistent_mini | `weave:///horangi/horangi4/object/KoHalluLens_NonExistent_mini:suMhzXfycG79qMYN3AjVQqGwtyFst1NFsbWdhk1jJTk` |
| BFCL_mini | `weave:///horangi/horangi4/object/BFCL_mini:ODywz9h7BWEfpYfAmkqjwLXQYxrsRWlPXCXNMoo3jTg` |
| KoMTBench_mini | `weave:///horangi/horangi4/object/KoMTBench_mini:GY9L798k1ezXyTlk7ILVZtAK0c3ii1ysPM7y1ahmCag` |
| SWEBench_Verified_80_mini | `weave:///horangi/horangi4/object/SWEBench_Verified_80_mini:AltUnANYMU9aYgmhrbKaKogRumY5eJt2lgECAbKax7w` |
| HAERAE_Bench_v1_mini | `weave:///horangi/horangi4/object/HAERAE_Bench_v1_mini:AUDj1Yc8irM87b4DOXS9LK31AXfCPo8Uh8aEXyGa9J4` |
| KoBALT_700_mini | `weave:///horangi/horangi4/object/KoBALT_700_mini:RXgDQTYja0ZySmuQhH0xRmEA36UJPH7YQcf1LrpD9o0` |

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
