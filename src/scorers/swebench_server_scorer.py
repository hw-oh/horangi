"""
SWE-bench Server Scorer

Scores resolved status by calling scoring server API instead of Docker-based official harness.

Patch processing pipeline:
1. extract_diff() - Extract diff from various formats (tags, code blocks)
2. repair_patch() - Use swebench official repair function
3. _fix_split_headers() - Fix broken header lines
4. extract_minimal_patch() - Minimize patch
5. LLM normalization - Last resort fallback for non-standard formats

References:
- Server implementation example: https://raw.githubusercontent.com/wandb/llm-leaderboard/main/scripts/server/swebench_server.py
- Evaluator implementation example: https://raw.githubusercontent.com/wandb/llm-leaderboard/main/scripts/evaluator/swe_bench.py

Environment variables:
- SWE_SERVER_URL: e.g., https://api.nejumi-swebench.org
- SWE_API_KEY: (optional) if server requires X-API-Key
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import ssl
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
)
from inspect_ai.solver import TaskState
from inspect_ai.model import get_model, ChatMessageUser

# Try to import swebench utilities (optional dependency)
try:
    from swebench.inference.make_datasets.utils import repair_patch as _swebench_repair_patch
    from swebench.inference.make_datasets.utils import extract_minimal_patch as _swebench_minimal_patch
    SWEBENCH_AVAILABLE = True
except ImportError:
    SWEBENCH_AVAILABLE = False
    _swebench_repair_patch = None
    _swebench_minimal_patch = None

logger = logging.getLogger(__name__)


# =============================================================================
# Prompt for LLM-based patch normalization (last resort fallback)
# =============================================================================
PATCH_NORMALIZE_PROMPT = """Convert the following patch to standard unified diff format.

IMPORTANT: Output ONLY the unified diff patch, with NO explanation, NO markdown code fences, NO additional text.

The output format MUST be exactly like this:
--- a/path/to/file.py
+++ b/path/to/file.py
@@ -start,count +start,count @@
 context line
-removed line
+added line

Input patch to convert:
{patch_text}

Output (unified diff only):"""


# =============================================================================
# Step 1: Extract diff from various formats
# =============================================================================
def extract_diff(response: str | None) -> str | None:
    """
    Extract diff from response formatted in different ways.
    
    Supports:
    - <diff>...</diff> or <patch>...</patch> tags
    - ```diff...``` or ```patch...``` code blocks
    - Other code blocks as fallback
    - Raw text after </s> token
    
    Reference: Official SWE-bench extract_diff function
    """
    if response is None:
        return None
    
    diff_matches = []
    other_matches = []
    
    # Pattern 1: XML-style tags <diff>...</diff> or <patch>...</patch>
    tag_pattern = re.compile(r"\<([\w-]+)\>(.*?)\<\/\1\>", re.DOTALL)
    for code, match in tag_pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    
    # Pattern 2: Markdown code blocks ```lang...```
    code_block_pattern = re.compile(r"```(\w+)?\n(.*?)```", re.DOTALL)
    for code, match in code_block_pattern.findall(response):
        if code in {"diff", "patch"}:
            diff_matches.append(match)
        else:
            other_matches.append(match)
    
    # Priority: diff/patch matches > other code blocks > raw response
    if diff_matches:
        return diff_matches[0]
    if other_matches:
        return other_matches[0]
    
    # Fallback: return everything before </s> token
    return response.split("</s>")[0]


# =============================================================================
# Step 2: Fix broken header lines
# =============================================================================
def _fix_split_headers(patch: str) -> str:
    """
    Join broken file header lines where path got split by newline.
    
    Example of broken header:
        --- a/astropy/modeling/separable.py
        +++ b/astropy
        /modeling/separable.py
    
    The second header line should be a single line.
    """
    if not patch:
        return patch
    
    lines: List[str] = patch.split("\n")
    fixed_lines: List[str] = []
    header_start = ("--- ", "+++ ")
    i = 0
    
    while i < len(lines):
        line = lines[i]
        if line.startswith(header_start):
            # Heuristic: if the line does not end with a known extension and the next
            # line continues the path (starts with '/') then join them.
            if not re.search(r"\.[a-zA-Z0-9]+$", line) and (i + 1) < len(lines):
                j = i + 1
                joined = line
                while j < len(lines):
                    next_line = lines[j]
                    if next_line.startswith(("@@ ", "diff ", "--- ", "+++ ")):
                        break
                    # Likely continuation of the path
                    joined += next_line.strip("\n")
                    j += 1
                fixed_lines.append(joined)
                i = j
                continue
        fixed_lines.append(line)
        i += 1
    
    return "\n".join(fixed_lines)


# =============================================================================
# Step 3: Check if patch is in standard format
# =============================================================================
def _is_standard_patch(text: str) -> bool:
    """
    Check if the text is in standard unified diff format.
    
    Standard formats:
    - diff --git a/file b/file
    - --- a/file / +++ b/file
    """
    if not text:
        return False
    
    # Check for standard unified diff patterns
    has_diff_git = "diff --git" in text
    has_unified_header = bool(re.search(r'^---\s+[ab]?/', text, re.MULTILINE))
    
    return has_diff_git or has_unified_header


# =============================================================================
# Step 4: Extract standard patch (for already-standard formats)
# =============================================================================
def _extract_standard_patch(text: str) -> str:
    """
    Extract patch from text that is already in standard unified diff format.
    """
    if not text:
        return ""

    # Case 1: diff --git header exists
    idx = text.find("diff --git")
    if idx != -1:
        patch_text = text[idx:]
        # Remove description text after diff (consider empty lines as separator)
        parts = re.split(r'\n\n(?=[A-Z])', patch_text, maxsplit=1)
        patch_text = parts[0]
        # Find where patch ends (where diff/---/+++/@@ patterns end)
        lines = patch_text.split('\n')
        valid_lines = []
        for line in lines:
            # Patch line patterns
            if (line.startswith('diff --git') or 
                line.startswith('---') or 
                line.startswith('+++') or 
                line.startswith('@@') or
                line.startswith('+') or
                line.startswith('-') or
                line.startswith(' ') or
                line == ''):
                valid_lines.append(line)
            elif valid_lines:  # If outside pattern after patch started, stop
                if not any(valid_lines[-1].startswith(p) for p in ['diff', '---', '+++', '@@', ' ', '+', '-']):
                    break
                valid_lines.append(line)
        return '\n'.join(valid_lines).strip() + '\n'

    # Case 2: unified diff starting with --- header
    m = re.search(r"(^---\s+[ab]?/?.+$)", text, re.MULTILINE)
    if m:
        idx = m.start()
        patch_text = text[idx:]
        # Remove description text after diff
        parts = re.split(r'\n\n(?=[A-Z])', patch_text, maxsplit=1)
        return parts[0].strip() + '\n'

    # Case 3: Treat entire content as patch (last fallback)
    return text.strip() + ('\n' if not text.endswith('\n') else '')


# =============================================================================
# Step 5: LLM-based normalization (last resort fallback)
# =============================================================================
async def _normalize_patch_with_llm(patch_text: str, normalizer_model: str) -> str:
    """
    Normalize non-standard patch format to unified diff using LLM.
    
    This is the LAST RESORT fallback when all other methods fail.
    Uses a cheap LLM to convert any format to standard unified diff.
    
    Args:
        patch_text: The raw patch text in any format
        normalizer_model: Model to use for normalization (e.g., "openai/gpt-5-mini")
    
    Returns:
        Unified diff format patch
    """
    try:
        model = get_model(normalizer_model)
        prompt = PATCH_NORMALIZE_PROMPT.format(patch_text=patch_text)
        
        response = await model.generate([
            ChatMessageUser(content=prompt)
        ])
        
        normalized = response.completion.strip()
        
        # Clean up: remove any code fences the LLM might have added
        normalized = extract_diff(normalized) or normalized
        
        # Validate: check if result looks like a valid patch
        if normalized and ("---" in normalized or "diff --git" in normalized):
            logger.info(f"[Patch Pipeline] LLM normalization successful using {normalizer_model}")
            return normalized + ('\n' if not normalized.endswith('\n') else '')
        else:
            logger.warning(f"[Patch Pipeline] LLM output doesn't look like a valid patch, using original")
            return patch_text
            
    except Exception as e:
        logger.warning(f"[Patch Pipeline] LLM normalization failed: {e}, using original")
        return patch_text


# =============================================================================
# Main Pipeline: Extract and normalize patch
# =============================================================================
async def _extract_and_normalize_patch(text: str, normalizer_model: str | None = None) -> str:
    """
    Extract and normalize patch from model output using a multi-step pipeline.
    
    Pipeline:
    1. extract_diff() - Extract from tags/code blocks
    2. repair_patch() - Use swebench official repair (if available)
    3. _fix_split_headers() - Fix broken header lines
    4. extract_minimal_patch() - Minimize patch (if available)
    5. LLM normalization - Last resort for non-standard formats
    
    Args:
        text: Raw model output
        normalizer_model: Optional model for LLM fallback normalization
    
    Returns:
        Extracted patch in unified diff format
    """
    if not text:
        return ""
    
    # Step 1: Extract diff from various formats (tags, code blocks, etc.)
    extracted = extract_diff(text)
    if not extracted:
        logger.warning("[Patch Pipeline] No diff found in response")
        return ""
    
    logger.debug(f"[Patch Pipeline] Step 1 (extract_diff): {len(extracted)} chars")
    
    # Step 2: Use swebench repair_patch if available
    repaired = extracted
    if SWEBENCH_AVAILABLE and _swebench_repair_patch:
        try:
            repaired = _swebench_repair_patch(extracted) or extracted
            logger.debug(f"[Patch Pipeline] Step 2 (repair_patch): {len(repaired)} chars")
        except Exception as e:
            logger.warning(f"[Patch Pipeline] repair_patch failed: {e}")
    
    # Step 3: Fix split headers
    fixed = _fix_split_headers(repaired) if repaired else ""
    logger.debug(f"[Patch Pipeline] Step 3 (_fix_split_headers): {len(fixed)} chars")
    
    # Step 4: Use swebench extract_minimal_patch if available
    minimal = fixed
    if SWEBENCH_AVAILABLE and _swebench_minimal_patch and fixed:
        try:
            minimal_result = _swebench_minimal_patch(fixed)
            # Use minimal if it's shorter and non-empty
            if minimal_result and len(minimal_result) <= len(fixed):
                minimal = minimal_result
                logger.debug(f"[Patch Pipeline] Step 4 (minimal_patch): {len(minimal)} chars")
        except Exception as e:
            logger.warning(f"[Patch Pipeline] extract_minimal_patch failed: {e}")
    
    # Check if we have a valid standard patch now
    if minimal and _is_standard_patch(minimal):
        final = _extract_standard_patch(minimal)
        # Ensure trailing newline
        if final and not final.endswith('\n'):
            final += '\n'
        logger.info(f"[Patch Pipeline] Success with standard processing: {len(final)} chars")
        return final
    
    # Step 5: LLM normalization as last resort
    if normalizer_model and minimal:
        logger.info(f"[Patch Pipeline] Non-standard format detected, trying LLM normalization with {normalizer_model}")
        normalized = await _normalize_patch_with_llm(minimal, normalizer_model)
        if normalized and _is_standard_patch(normalized):
            return normalized
    
    # Final fallback: return whatever we have
    if minimal:
        logger.warning("[Patch Pipeline] Returning non-standard patch as-is")
        return minimal.strip() + ('\n' if not minimal.endswith('\n') else '')
    
    logger.warning("[Patch Pipeline] No usable patch found")
    return ""


def _http_json(
    method: str,
    url: str,
    headers: Optional[Dict[str, str]] = None,
    body: Optional[dict] = None,
    timeout: int = 60,
) -> dict:
    """
    HTTP request helper (SSL verification bypass + retry logic)
    
    Reference: https://github.com/wandb/llm-leaderboard/blob/main/scripts/evaluator/swe_bench.py
    """
    data = None
    req_headers = dict(headers) if headers else {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req_headers["Content-Type"] = "application/json"

    # Bypass SSL certificate verification (for development)
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    # No retry for POST (has side-effects)
    max_attempts = 1 if method.upper() == "POST" else 3
    attempt = 0
    backoff_sec = 2.0
    last_err: Optional[Exception] = None

    while attempt < max_attempts:
        attempt += 1
        try:
            req = Request(url=url, data=data, headers=req_headers, method=method.upper())
            with urlopen(req, timeout=timeout, context=ssl_context) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else {}
        except (TimeoutError, URLError, HTTPError) as e:
            # Raise immediately for unrecoverable HTTP errors
            if isinstance(e, HTTPError) and e.code not in [524, 502, 503, 504]:
                raise
            last_err = e
            if attempt >= max_attempts:
                raise
            logger.warning(
                f"[API] {method} request failed (attempt {attempt}/{max_attempts}): {e}. "
                f"Retrying in {backoff_sec}s..."
            )
            time.sleep(backoff_sec)
            backoff_sec *= 2

    if last_err:
        raise last_err
    return {}


def _run_job_sync(
    *,
    server_url: str,
    api_key: Optional[str],
    instance_id: str,
    patch_diff: str,
    timeout_sec: int,
    poll_interval_sec: float,
    namespace: str,
    tag: str,
    model_name_or_path: str,
) -> dict[str, Any]:
    # Add header only if API key exists (empty dict otherwise)
    headers: Optional[dict[str, str]] = {"X-API-Key": api_key} if api_key else None

    # 1) create job
    create = _http_json(
        "POST",
        f"{server_url}/v1/jobs",
        headers=headers,
        body={
            "instance_id": instance_id,
            "patch_diff": patch_diff,
            "namespace": namespace,
            "tag": tag,
            "timeout_sec": timeout_sec,
            "model_name_or_path": model_name_or_path,
        },
        timeout=300,
    )
    job_id = create.get("job_id")
    if not job_id:
        raise RuntimeError(f"Invalid server response (missing job_id): {create}")

    # 2) poll status
    start = time.time()
    last = create
    while True:
        if time.time() - start > (timeout_sec + 120):
            raise TimeoutError(f"Job polling timed out: {job_id}")

        status = _http_json("GET", f"{server_url}/v1/jobs/{job_id}", headers=headers, timeout=300)
        last = status
        s = status.get("status")
        if s in ("finished", "failed"):
            break
        time.sleep(poll_interval_sec)

    # 3) fetch report (best-effort)
    report: Optional[dict] = None
    try:
        report = _http_json("GET", f"{server_url}/v1/jobs/{job_id}/report", headers=headers, timeout=300)
    except Exception:
        report = None

    return {"job_id": job_id, "status": last, "report": report}


@scorer(metrics=[accuracy()])
def swebench_server_scorer(
    server_url: str | None = None,
    api_key: str | None = None,
    timeout_sec: int = 1800,
    poll_interval_sec: float = 2.0,
    namespace: str = "swebench",
    tag: str = "latest",
    model_name_or_path: str = "horangi",
    normalizer_model: str | None = "openai/gpt-5-mini",
) -> Scorer:
    """
    SWE-bench Server Scorer
    
    Patch processing pipeline:
    1. extract_diff() - Extract from tags/code blocks
    2. repair_patch() - swebench official repair (if swebench installed)
    3. _fix_split_headers() - Fix broken header lines
    4. extract_minimal_patch() - Minimize (if swebench installed)
    5. LLM normalization - Last resort fallback
    
    Args:
        server_url: SWE-bench evaluation server URL
        api_key: API key for the server
        timeout_sec: Timeout for evaluation job
        poll_interval_sec: Polling interval for job status
        namespace: Namespace for the evaluation
        tag: Tag for the evaluation
        model_name_or_path: Model name to log
        normalizer_model: Model for LLM fallback normalization.
                         Set to None to disable LLM fallback.
                         Default: "openai/gpt-5-mini"
    """
    async def score(state: TaskState, target: Target) -> Score:
        server = server_url or os.getenv("SWE_SERVER_URL", "https://api.nejumi-swebench.org")
        key = api_key or os.getenv("SWE_API_KEY")

        metadata = state.metadata or {}
        instance_id = (
            metadata.get("instance_id")
            or state.sample_id
            or metadata.get("id")
            or ""
        )

        response = state.output.completion if state.output else ""
        patch = await _extract_and_normalize_patch(response, normalizer_model)

        if not instance_id or not patch.strip():
            return Score(
                value=INCORRECT,
                answer=(response or ""),
                explanation="Missing instance_id or empty patch",
                metadata={
                    "judgment": "ERROR",
                    "instance_id": instance_id,
                    "has_patch": bool(patch.strip()),
                },
            )

        # Offload network I/O to thread
        try:
            result = await asyncio.to_thread(
                _run_job_sync,
                server_url=server.rstrip("/"),
                api_key=key,
                instance_id=str(instance_id),
                patch_diff=patch,
                timeout_sec=timeout_sec,
                poll_interval_sec=poll_interval_sec,
                namespace=namespace,
                tag=tag,
                model_name_or_path=model_name_or_path,
            )
        except HTTPError as e:
            return Score(
                value=INCORRECT,
                answer=patch,
                explanation=f"HTTPError from server: {e}",
                metadata={"judgment": "ERROR", "instance_id": instance_id},
            )
        except URLError as e:
            return Score(
                value=INCORRECT,
                answer=patch,
                explanation=f"URLError from server: {e}",
                metadata={"judgment": "ERROR", "instance_id": instance_id},
            )
        except Exception as e:
            return Score(
                value=INCORRECT,
                answer=patch,
                explanation=f"Failed to score via server: {e}",
                metadata={"judgment": "ERROR", "instance_id": instance_id},
            )

        report = (result or {}).get("report") or {}
        resolved_ids = set(report.get("resolved_ids", []) or [])
        is_resolved = str(instance_id) in resolved_ids

        return Score(
            value=CORRECT if is_resolved else INCORRECT,
            answer=patch,
            explanation="resolved" if is_resolved else "not resolved",
            metadata={
                "judgment": "CORRECT" if is_resolved else "INCORRECT",
                "instance_id": instance_id,
                "job_id": result.get("job_id"),
                "server_status": (result.get("status") or {}).get("status"),
                "report_summary": {
                    "resolved_ids_len": len(resolved_ids),
                    "unresolved_ids_len": len(report.get("unresolved_ids", []) or []),
                },
            },
        )

    return score
