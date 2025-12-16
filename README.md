# 🐯 Horangi - 한국어 LLM 벤치마크 평가 프레임워크

**호랑이(Horangi)**는 한국어 LLM의 성능을 종합적으로 평가하는 오픈소스 벤치마크 프레임워크입니다.

[WandB/Weave](https://wandb.ai/site/weave)와 [Inspect AI](https://inspect.ai-safety-institute.org.uk/)를 통합하여 **범용언어성능(GLP)**과 **가치정렬성능(ALT)** 두 축으로 한국어 LLM을 평가합니다.

<div align="center">

🏆 **[호랑이 리더보드](https://horangi.ai)** - 한국어 LLM 성능 순위 확인

</div>

- **범용언어성능 (GLP)**: 언어 이해, 지식, 추론, 코딩, 함수호출 등 15개 벤치마크
- **가치정렬성능 (ALT)**: 제어성, 윤리, 유해성/편향성 방지, 환각 방지 등 8개 벤치마크

### 📬 문의

| | |
|---|---|
| 리더보드 등재 신청 | [신청 폼](https://docs.google.com/forms/d/e/1FAIpQLSdQERNX8jCEuqzUiodjnUdAI7JRCemy5sgmVylio-u0DRb9Xw/viewform) |
| 일반 문의 | contact-kr@wandb.com |

---

## ✨ 특징

- 🇰🇷 **20여개 한국어 벤치마크** 지원
- 📊 **WandB/Weave 자동 로깅** - 실험 추적 및 결과 비교
- 🚀 **다양한 모델 지원** - OpenAI, Claude, Gemini, Solar, EXAONE 등
- 🛠️ **CLI 지원** - `horangi` 명령어로 간편 실행

## 📦 설치

```bash
# uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 저장소 클론
git clone https://github.com/wandb-korea/horangi.git
cd horangi

# 의존성 설치
uv sync
```

### 환경 변수 설정

```bash
# 필수
export WANDB_API_KEY=your_wandb_api_key

# 모델별 API 키 (사용할 모델에 따라)
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
export UPSTAGE_API_KEY=your_upstage_api_key
```

## 🚀 빠른 시작

### 기본 사용법 (CLI)

```bash
# 벤치마크 목록 확인
uv run horangi --list

# 벤치마크 실행
uv run horangi kmmlu --model openai/gpt-4o

# 샘플 수 제한
uv run horangi kmmlu --model openai/gpt-4o -T limit=10
```

### 다양한 모델 사용

```bash
# OpenAI
uv run horangi kmmlu --model openai/gpt-4o

# Anthropic
uv run horangi kmmlu --model anthropic/claude-3-5-sonnet-20241022

# Google
uv run horangi kmmlu --model google/gemini-1.5-pro

# vLLM (로컬)
uv run horangi kmmlu --model vllm/LGAI-EXAONE/EXAONE-3.5-32B-Instruct

# Ollama (로컬)
uv run horangi kmmlu --model ollama/llama3.1:70b
```

### OpenAI 호환 API (Solar, Grok 등)

OpenAI 호환 API를 사용하는 모델은 `base_url`과 `api_key`를 직접 지정합니다:

```bash
uv run horangi kmmlu \
  --model openai/solar-pro2 \
  --model-base-url https://api.upstage.ai/v1 \
  -M api_key=$UPSTAGE_API_KEY
```

---

## ⚙️ 모델 설정 파일 (선택)

자주 사용하는 모델이나 복잡한 설정은 **config 파일**로 관리할 수 있습니다.

### config 사용

```bash
# 모델 설정 목록 확인
uv run horangi --list-models

# config로 실행
uv run horangi kmmlu --config gpt-4o
uv run horangi kmmlu --config solar_pro2
```

### 새 모델 config 추가

```bash
# 1. 템플릿 복사
cp configs/models/_template.yaml configs/models/my-model.yaml

# 2. 설정 편집
```

```yaml
# configs/models/my-model.yaml

# 모델 ID (표시용)
model_id: upstage/solar-pro2

# OpenAI 호환 API 사용 시
api_provider: openai

# API 설정
base_url: https://api.upstage.ai/v1
api_key_env: UPSTAGE_API_KEY

# 기본 파라미터
defaults:
  temperature: 0.0
  max_tokens: 4096

# 벤치마크별 오버라이드 (선택)
benchmarks:
  bfcl:
    use_native_tools: true
```

```bash
# 3. 실행
uv run horangi kmmlu --config my-model
```

### `--model` vs `--config`

| 방식 | 사용 시점 |
|------|----------|
| `--model` | 간단한 실행, 일회성 테스트 |
| `--config` | 반복 사용, OpenAI 호환 API, 벤치마크별 설정 필요 시 |

---

## 📊 지원 벤치마크

| 대분류 | 평가 영역 | 벤치마크 | 설명 | 샘플개수 |
|--------|----------|----------|------|-----:|
| **범용언어성능 (GLP)** | 구문해석 | `ko_balt_700_syntax` | 문장 구조 분석, 문법적 타당성 평가 | 100 |
| | 의미해석 | `ko_balt_700_semantic` | 문맥 기반 추론, 의미적 일관성 평가 | 100 |
| | 의미해석 | `haerae_bench_v1_rc` | 독해 기반 의미 해석력 평가 | 100 |
| | 표현 | `mtbench_ko` | 글쓰기, 역할극, 인문학적 표현력 (LLM Judge) | 80 |
| | 정보검색 | `squad_kor_v1` | 질의응답 기반 정보검색 능력 | 100 |
| | 일반지식 | `kmmlu` | 상식, STEM 기초학문 이해도 | 100 |
| | 일반지식 | `haerae_bench_v1_wo_rc` | 멀티턴 질의응답 기반 지식 평가 | 100 |
| | 전문지식 | `kmmlu_pro` | 의학, 법률, 공학 등 고난도 전문지식 | 100 |
| | 전문지식 | `ko_hle` | 한국어 고난도 전문가 수준 문제 | 100 |
| | 상식추론 | `ko_hellaswag` | 문장 완성, 다음 문장 예측 | 100 |
| | 수학추론 | `ko_gsm8k` | 수학 문제 풀이 | 100 |
| | 수학추론 | `ko_aime2025` | AIME 2025 고난도 수학 | 30 |
| | 추상추론 | `ko_arc_agi` | 시각적/구조적 추론, 추상적 문제 해결 | 100 |
| | 코딩 | `swebench_verified_official_80` | GitHub 이슈 해결 능력 | 80 |
| | 함수호출 | `bfcl` | 함수 호출 정확성 (단일, 멀티턴, 무관계검출) | 258 |
| **가치정렬성능 (ALT)** | 제어성 | `ifeval_ko` | 지시문 수행, 명령 준수 능력 | 100 |
| | 윤리/도덕 | `ko_moral` | 사회 규범 준수, 안전한 언어 생성 | 100 |
| | 유해성방지 | `korean_hate_speech` | 혐오발언, 공격적 발화 탐지 및 억제 | 100 |
| | 편향성방지 | `kobbq` | 특정 집단/속성에 대한 편향성 평가 | 100 |
| | 환각방지 | `ko_truthful_qa` | 사실성 검증, 근거 기반 답변 생성 | 100 |
| | 환각방지 | `ko_hallulens_wikiqa` | Wikipedia QA 기반 환각 평가 | 100 |
| | 환각방지 | `ko_hallulens_longwiki` | 긴 문맥 Wikipedia 환각 평가 | 100 |
| | 환각방지 | `ko_hallulens_nonexistent` | 가상 엔티티 거부 능력 평가 | 100 |
| | | **총합** | | **~2,348** |

<details>
<summary>📦 데이터셋 참조 (Weave)</summary>

데이터셋은 `horangi/horangi4` 프로젝트에 업로드되어 있습니다:

| 데이터셋 | Weave Ref |
|----------|-----------|
| KoHellaSwag_mini | `weave:///horangi/horangi4/object/KoHellaSwag_mini:latest` |
| KoAIME2025_mini | `weave:///horangi/horangi4/object/KoAIME2025_mini:latest` |
| IFEval_Ko_mini | `weave:///horangi/horangi4/object/IFEval_Ko_mini:latest` |
| HAERAE_Bench_v1_mini | `weave:///horangi/horangi4/object/HAERAE_Bench_v1_mini:latest` |
| KoBALT_700_mini | `weave:///horangi/horangi4/object/KoBALT_700_mini:latest` |
| KMMLU_mini | `weave:///horangi/horangi4/object/KMMLU_mini:latest` |
| KMMLU_Pro_mini | `weave:///horangi/horangi4/object/KMMLU_Pro_mini:latest` |
| SQuAD_Kor_v1_mini | `weave:///horangi/horangi4/object/SQuAD_Kor_v1_mini:latest` |
| KoTruthfulQA_mini | `weave:///horangi/horangi4/object/KoTruthfulQA_mini:latest` |
| KoMoral_mini | `weave:///horangi/horangi4/object/KoMoral_mini:latest` |
| KoARC_AGI_mini | `weave:///horangi/horangi4/object/KoARC_AGI_mini:latest` |
| KoGSM8K_mini | `weave:///horangi/horangi4/object/KoGSM8K_mini:latest` |
| KoreanHateSpeech_mini | `weave:///horangi/horangi4/object/KoreanHateSpeech_mini:latest` |
| KoBBQ_mini | `weave:///horangi/horangi4/object/KoBBQ_mini:latest` |
| KoHLE_mini | `weave:///horangi/horangi4/object/KoHLE_mini:latest` |
| KoHalluLens_WikiQA_mini | `weave:///horangi/horangi4/object/KoHalluLens_WikiQA_mini:latest` |
| KoHalluLens_LongWiki_mini | `weave:///horangi/horangi4/object/KoHalluLens_LongWiki_mini:latest` |
| KoHalluLens_NonExistent_mini | `weave:///horangi/horangi4/object/KoHalluLens_NonExistent_mini:latest` |
| BFCL_mini | `weave:///horangi/horangi4/object/BFCL_mini:latest` |
| KoMTBench_mini | `weave:///horangi/horangi4/object/KoMTBench_mini:latest` |
| SWEBench_Verified_80_mini | `weave:///horangi/horangi4/object/SWEBench_Verified_80_mini:latest` |

</details>

---

## 📈 결과 확인

### Weave Evaluation

Weave UI에서 상세 결과 확인:
- 샘플별 점수 및 응답
- 모델 간 비교
- 집계 메트릭 (Scores 섹션)

### Weave Leaderboard (모델 비교)

여러 모델의 평가 결과를 Weave UI의 **Leaderboard**로 비교할 수 있습니다.

```bash
# Leaderboard 생성/업데이트
uv run horangi leaderboard --project horangi/horangi4
```

---

## 📁 프로젝트 구조

```
horangi/
├── horangi.py              # @task 함수 정의 (진입점)
├── configs/
│   └── models/             # 모델 설정 파일
├── src/
│   ├── benchmarks/         # 벤치마크 설정
│   ├── core/               # 핵심 로직
│   ├── scorers/            # 커스텀 Scorer
│   ├── solvers/            # 커스텀 Solver
│   └── cli/                # CLI 엔트리포인트
└── create_benchmark/       # 데이터셋 생성 스크립트
```

> 📖 **새 벤치마크 추가 방법**은 [src/README.md](src/README.md)를 참고하세요.

---

## 📚 참고 자료

- [Inspect AI Documentation](https://inspect.ai-safety-institute.org.uk/)
- [inspect-wandb (fork)](https://github.com/hw-oh/inspect_wandb)
- [inspect_evals](https://github.com/UKGovernmentBEIS/inspect_evals)
- [WandB Weave](https://wandb.ai/site/weave)

## 📄 라이선스

MIT License
