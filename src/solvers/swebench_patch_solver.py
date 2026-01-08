"""
SWE-bench Patch Solver

Generate a unified diff patch from a problem statement (SWE-bench style).

References:
- Japanese SWE-bench prompt format (0.21 pass rate)
- https://github.com/wandb/llm-leaderboard/blob/main/scripts/evaluator/swe_bench.py
"""

from __future__ import annotations

import re
from inspect_ai.model import ChatMessageUser
from inspect_ai.solver import Generate, Solver, TaskState, solver


PATCH_EXAMPLE = '''Here is an example of a patch file. It consists of changes to the code base. It specifies the file names, the line numbers of each change, and the removed and added lines. A single patch file can contain changes to multiple files.

<patch>
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
 
 
 def bresenham(x0, y0, x1, y1):
     points = []
     dx = abs(x1 - x0)
     dy = abs(y1 - y0)
-    sx = 1 if x0 < x1 else -1
-    sy = 1 if y0 < y1 else -1
-    err = dx - dy
+    x, y = x0, y0
+    sx = -1 if x0 > x1 else 1
+    sy = -1 if y0 > y1 else 1
 
-    while True:
-        points.append((x0, y0))
-        if x0 == x1 and y0 == y1:
-            break
-        e2 = 2 * err
-        if e2 > -dy:
+    if dx > dy:
+        err = dx / 2.0
+        while x != x1:
+            points.append((x, y))
             err -= dy
-            x0 += sx
-        if e2 < dx:
-            err += dx
-            y0 += sy
+            if err < 0:
+                y += sy
+                err += dx
+            x += sx
+    else:
+        err = dy / 2.0
+        while y != y1:
+            points.append((x, y))
+            err -= dx
+            if err < 0:
+                x += sx
+                err += dy
+            y += sy
 
+    points.append((x, y))
     return points
</patch>

CRITICAL: When generating the patch, you MUST include proper line numbers in the @@ hunk headers. Each hunk header must follow the format @@ -start,count +start,count @@ where start is the line number and count is the number of lines. Do NOT use @@ @@ without line numbers.
'''


def _extract_target_files_from_test_patch(test_patch: str) -> list[str]:
    """Extract test file paths from test_patch (used as hints for modules to modify)"""
    if not test_patch:
        return []
    # Pattern: diff --git a/path/to/file.py b/path/to/file.py
    files = re.findall(r'diff --git a/(\S+)', test_patch)
    return list(set(files))


def _extract_fail_to_pass_info(fail_to_pass: str | list) -> str:
    """Extract test info from FAIL_TO_PASS"""
    if not fail_to_pass:
        return ""
    if isinstance(fail_to_pass, str):
        try:
            import json
            fail_to_pass = json.loads(fail_to_pass)
        except:
            return fail_to_pass
    if isinstance(fail_to_pass, list):
        return "\n".join(f"- {t}" for t in fail_to_pass[:5])  # Max 5 items
    return str(fail_to_pass)


@solver
def swebench_patch_solver(include_hints: bool = True) -> Solver:
    """
    SWE-bench Patch Solver (Japanese SWE-bench style)
    
    Generates a unified diff patch from:
    - state.input: problem_statement
    - metadata: repo, instance_id, hints_text, test_patch, FAIL_TO_PASS, code, etc.
    
    If 'code' field is present in metadata (from Japanese dataset), it will be included
    in the prompt for better accuracy.
    
    Args:
        include_hints: Whether to include hints_text in the prompt. Default: True.
                      Set to False to exclude hints for harder evaluation.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        metadata = state.metadata or {}

        problem = state.input if isinstance(state.input, str) else str(state.input)
        hints = metadata.get("hints_text", "") or ""
        repo = metadata.get("repo", "")
        version = metadata.get("version", "")
        test_patch = metadata.get("test_patch", "") or ""
        fail_to_pass = metadata.get("FAIL_TO_PASS", "") or ""
        code = metadata.get("code", "") or ""  # Actual code file content

        # Extract related module path hints from test files
        test_files = _extract_target_files_from_test_patch(test_patch)
        fail_info = _extract_fail_to_pass_info(fail_to_pass)

        # Build the user prompt (Japanese SWE-bench style - proven 0.21 pass rate)
        user_prompt = "You will be provided with a partial code base and an issue statement explaining a problem to resolve.\n\n"
        
        # Issue section
        user_prompt += f"<issue>\n{problem}\n</issue>\n\n"

        # CODE SECTION - Key! Actual code file content
        if code.strip():
            user_prompt += f"<code>\n{code}\n</code>\n\n"
        else:
            # If code is not available, use existing hints
            # Repository info
            if repo:
                user_prompt += f"<repository>\n{repo}"
                if version:
                    user_prompt += f" (version {version})"
                user_prompt += "\n</repository>\n\n"

            # Test files hint (helps identify which module to modify)
            if test_files:
                user_prompt += "<test_files>\n"
                user_prompt += "The following test files are related to this issue:\n"
                for tf in test_files:
                    user_prompt += f"- {tf}\n"
                user_prompt += "</test_files>\n\n"

            # Tests that should pass after fix
            if fail_info:
                user_prompt += "<tests_to_fix>\n"
                user_prompt += "The following tests are currently failing and should pass after your fix:\n"
                user_prompt += fail_info
                user_prompt += "\n</tests_to_fix>\n\n"

        # Hints section (if available and enabled)
        if include_hints and hints.strip():
            user_prompt += f"<hints>\n{hints}\n</hints>\n\n"

        # Patch example
        user_prompt += PATCH_EXAMPLE
        user_prompt += "\n"

        # Final instruction
        user_prompt += (
            "I need you to solve the provided issue by generating a single patch file that I can apply directly to this repository using git apply. Please respond with a single patch file in the format shown above.\n\n"
            "Wrap your patch in <patch> tags like this: <patch>your patch here</patch>\n\n"
            "Respond below:\n"
        )

        state.messages = [ChatMessageUser(content=user_prompt)]
        return await generate(state)

    return solve
