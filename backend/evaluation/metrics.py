from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from typing import Set

class ExactMatchMetric(BaseMetric):
    """
    Custom DeepEval metric that checks if the generated output matches expected output exactly.
    """
    def __init__(self, threshold: float = 1.0, case_sensitive: bool = False):
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        if not self.case_sensitive:
            actual = actual.strip().lower()
            expected = expected.strip().lower()
        else:
            actual = actual.strip()
            expected = expected.strip()

        if actual == expected:
            self.score = 1.0
            self.reason = "Actual output matches expected output exactly."
        else:
            self.score = 0.0
            self.reason = f"Actual output does not match expected output."

        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self) -> str:
        return "Exact Match"


class JaccardSimilarityMetric(BaseMetric):
    """
    Custom DeepEval metric that calculates the Jaccard similarity (word overlap)
    between the actual output and the expected output.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""

    def measure(self, test_case: LLMTestCase) -> float:
        actual = test_case.actual_output or ""
        expected = test_case.expected_output or ""

        actual_words: Set[str] = set(actual.strip().lower().split())
        expected_words: Set[str] = set(expected.strip().lower().split())

        if not actual_words and not expected_words:
            self.score = 1.0
            self.reason = "Both generated and expected responses are empty."
            return self.score

        intersection = actual_words.intersection(expected_words)
        union = actual_words.union(expected_words)

        self.score = float(len(intersection)) / len(union)
        self.reason = f"Jaccard index calculated at {self.score:.2f} based on word token overlap."
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.score >= self.threshold

    @property
    def __name__(self) -> str:
        return "Jaccard Similarity"
