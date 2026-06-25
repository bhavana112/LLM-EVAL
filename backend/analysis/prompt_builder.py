from typing import List, Dict, Any, Optional

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an expert AI Quality Engineer and ML Infrastructure Engineer.\n"
    "Your task is to analyze failed test cases from an LLM evaluation experiment, "
    "group them into failure categories, explain recurring failure patterns, "
    "and provide actionable, practical recommendations to improve prompt templates, "
    "retrieval quality, or model configurations."
)

DEFAULT_USER_TEMPLATE = """
We have run an evaluation experiment and identified {num_failures} failed test cases out of {total_cases} total test cases.
Below are details of the failed test cases.

Allowed Failure Categories:
{allowed_categories}

Failed Test Case Log:
{failed_cases_log}

Please perform the following analysis:
1. Classify each failure into one of the allowed categories.
2. Group similar failures within the same category into logical thematic groups (GroupedFailure). For each group, list the corresponding test case IDs and a short explanation of the failure theme.
3. Count the number of failed cases classified in each category (category_counts).
4. Identify 1 to 5 high-level recurring failure patterns (identified_patterns).
5. Generate a concise, high-level summary of the overall findings (ai_generated_summary).
6. Provide 2 to 5 actionable, concrete recommendations for how to improve the LLM system configurations, system prompt, or retrieval strategies (recommendations).

You MUST output your response ONLY as a valid JSON object matching the following structure:
{{
  "detected_categories": ["Category A", "Category B"],
  "grouped_failures": {{
    "Category A": [
      {{
        "test_case_ids": ["case_id_1", "case_id_2"],
        "description": "Short explanation of the theme of failures"
      }}
    ]
  }},
  "category_counts": {{
    "Category A": 2
  }},
  "identified_patterns": [
    "Explanation of pattern 1",
    "Explanation of pattern 2"
  ],
  "ai_generated_summary": "Concise high-level summary...",
  "recommendations": [
    "Practical recommendation 1",
    "Practical recommendation 2"
  ]
}}

Ensure that all returned keys are present and follow this schema exactly. Do not include any other conversational text or surrounding explanations. Output ONLY the JSON block.
"""

class PromptBuilder:
    """Responsible for generating structured prompts for failure analysis."""

    def __init__(
        self,
        system_instruction: str = DEFAULT_SYSTEM_INSTRUCTION,
        user_template: str = DEFAULT_USER_TEMPLATE,
        max_content_len: int = 500
    ):
        """
        system_instruction: The system prompt instructing the LLM on its role.
        user_template: The template string for building the user prompt.
        max_content_len: Max character length allowed for text outputs (prompt, expected, generated)
                          before truncation to minimize token usage.
        """
        self.system_instruction = system_instruction
        self.user_template = user_template
        self.max_content_len = max_content_len

    def _truncate(self, text: Optional[str]) -> str:
        if not text:
            return "N/A"
        if len(text) <= self.max_content_len:
            return text
        return text[:self.max_content_len] + "... [TRUNCATED]"

    def format_failed_case(self, case: Dict[str, Any]) -> str:
        """Formats a single failed case into a clean, token-efficient string representation."""
        test_case_id = case.get("test_case_id", "unknown")
        prompt = self._truncate(case.get("prompt", ""))
        expected = self._truncate(case.get("expected_output", ""))
        generated = self._truncate(case.get("generated_output", ""))
        scores = case.get("scores", {})
        
        # Determine failure reason
        error_msg = case.get("error_message")
        if error_msg:
            reason = f"Execution/Evaluation error: {error_msg}"
        else:
            failed_metrics = [f"{m}={v:.2f}" for m, v in scores.items() if v < 1.0]
            reason = f"Failed metrics: {', '.join(failed_metrics)}" if failed_metrics else "Evaluation metrics fell below threshold."

        formatted_scores = ", ".join([f"{k}: {v:.2f}" for k, v in scores.items()])
        
        return (
            f"--- Test Case ID: {test_case_id} ---\n"
            f"Prompt: {prompt}\n"
            f"Expected Output: {expected}\n"
            f"Generated Output: {generated}\n"
            f"Metric Scores: {formatted_scores if formatted_scores else 'None'}\n"
            f"Failure Reason: {reason}\n"
        )

    def build_prompt(
        self,
        failed_cases: List[Dict[str, Any]],
        total_cases: int,
        allowed_categories: List[str]
    ) -> str:
        """
        Constructs the complete user prompt with formatted failed cases and allowed categories.
        """
        case_logs = []
        for case in failed_cases:
            case_logs.append(self.format_failed_case(case))
            
        failed_cases_log = "\n".join(case_logs)
        
        formatted_categories = ", ".join(allowed_categories)
        
        return self.user_template.format(
            num_failures=len(failed_cases),
            total_cases=total_cases,
            allowed_categories=formatted_categories,
            failed_cases_log=failed_cases_log
        )
