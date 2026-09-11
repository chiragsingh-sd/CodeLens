"""
LLM-based Evaluation Judge for CodeLens Generation Quality (Step 3).

Evaluates generated answers along three standardized dimensions:
1. Answer Correctness (vs. Ground-Truth Reference Answer) [0-2]
2. Faithfulness / Groundedness (vs. Retrieved Context) [0-2]
3. Relevance (vs. Developer Question) [0-2]
4. Detection of Unsupported / Hallucinated Claims

Includes deterministic diagnostic checks for expected symbol and file mentions.
"""

import json
from pathlib import Path
import re
import sys
import time
from typing import Any

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.core.llm import call_llm


JUDGE_SYSTEM_PROMPT = """You are an expert, objective software engineering evaluation judge.
Your task is to evaluate a generated answer to a repository-level technical question according to a strict rubric.

You will receive:
1. QUESTION: The developer's query about the repository.
2. REFERENCE ANSWER: The verified technical ground-truth answer.
3. RETRIEVED CONTEXT: The repository code snippets provided to the assistant.
4. GENERATED ANSWER: The answer produced by the assistant.

Evaluate along three dimensions using this exact 0-2 integer scale:

DIMENSION 1: CORRECTNESS (vs. Reference Answer)
- 0: Incorrect. Fails to answer the question correctly or makes factually wrong technical assertions.
- 1: Partially correct / incomplete. Mentions some correct aspects but omits key components or is overly vague.
- 2: Correct. Accurate, complete, and agrees with the ground-truth reference answer.

DIMENSION 2: FAITHFULNESS / GROUNDEDNESS (vs. Retrieved Context)
- 0: Substantially unsupported. Makes major technical claims about the codebase that cannot be found in or derived from the retrieved context.
- 1: Partially supported / contains unsupported claims. Most claims are supported, but introduces non-existent functions, fabricated parameters, or unverified claims.
- 2: Fully supported. Every factual claim is directly verifiable from the provided retrieved context.

DIMENSION 3: RELEVANCE (vs. Question)
- 0: Irrelevant. Fails to address the question or goes completely off-topic.
- 1: Partially relevant. Addresses the topic tangentially or answers only a minor part of the question.
- 2: Directly answers the question. Directly, concisely, and specifically addresses what was asked.

UNSUPPORTED CLAIMS:
- Set unsupported_claims to true if ANY claim in the generated answer cannot be verified from the retrieved context.
- If true, list the specific unsupported/hallucinated claims in unsupported_claim_details.
- If false, unsupported_claim_details must be an empty list [].

Output strictly valid JSON with this exact schema:
{
  "correctness": 0 | 1 | 2,
  "faithfulness": 0 | 1 | 2,
  "relevance": 0 | 1 | 2,
  "unsupported_claims": true | false,
  "unsupported_claim_details": ["<claim 1>", ...],
  "reason": "<concise explanation justifying the scores>"
}"""


def parse_judge_response(raw_response: str) -> dict[str, Any]:
    """
    Cleans and validates the JSON output from the LLM judge.
    """
    clean_text = raw_response.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?", "", clean_text)
        clean_text = re.sub(r"```$", "", clean_text).strip()

    try:
        data = json.loads(clean_text)
    except Exception:
        match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
        else:
            raise ValueError(f"Failed to parse JSON from judge response: {raw_response[:200]}")

    # Validate and clamp scores
    correctness = max(0, min(2, int(data.get("correctness", 0))))
    faithfulness = max(0, min(2, int(data.get("faithfulness", 0))))
    relevance = max(0, min(2, int(data.get("relevance", 0))))

    unsupported_claims = bool(data.get("unsupported_claims", False))
    details = data.get("unsupported_claim_details", [])
    if not isinstance(details, list):
        details = [str(details)] if details else []

    reason = str(data.get("reason", "")).strip()

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "relevance": relevance,
        "unsupported_claims": unsupported_claims,
        "unsupported_claim_details": details,
        "reason": reason,
    }


def evaluate_answer(
    question: str,
    reference_answer: str,
    retrieved_context: str,
    generated_answer: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Invokes the LLM Judge to evaluate a generated answer against
    the reference answer and retrieved context.
    Retries with backoff if transient errors occur.
    """
    user_prompt = f"""QUESTION:
{question}

REFERENCE ANSWER:
{reference_answer}

RETRIEVED CONTEXT:
{retrieved_context}

GENERATED ANSWER:
{generated_answer}
"""
    last_err = None
    for attempt in range(max_retries):
        try:
            raw_judge_output = call_llm(
                JUDGE_SYSTEM_PROMPT,
                user_prompt,
                json_mode=True,
            )
            parsed = parse_judge_response(raw_judge_output)
            parsed["evaluator_status"] = "success"
            return parsed
        except Exception as e:
            last_err = e
            time.sleep(2 * (attempt + 1))

    return {
        "correctness": 0,
        "faithfulness": 0,
        "relevance": 0,
        "unsupported_claims": False,
        "unsupported_claim_details": [],
        "reason": f"Evaluator failure after {max_retries} attempts: {last_err}",
        "evaluator_status": "failed",
    }


def run_deterministic_checks(generated_answer: str, ground_truth: dict[str, Any]) -> dict[str, Any]:
    """
    Performs keyword/symbol presence checks as diagnostic signals.
    Does not substitute for evaluation scoring.
    """
    ans_lower = generated_answer.lower()

    expected_symbols = [
        ps["symbol"].strip()
        for ps in ground_truth.get("primary_symbols", [])
    ]
    expected_files = [
        f.replace("\\", "/").strip()
        for f in ground_truth.get("acceptable_files", [])
    ]

    symbols_found = [
        sym for sym in expected_symbols
        if sym.lower() in ans_lower
    ]
    files_found = [
        f for f in expected_files
        if Path(f).name.lower() in ans_lower or f.lower() in ans_lower
    ]

    return {
        "expected_symbols": expected_symbols,
        "symbols_found": symbols_found,
        "has_symbol_mention": len(symbols_found) > 0,
        "expected_files": expected_files,
        "files_found": files_found,
        "has_file_mention": len(files_found) > 0,
    }
