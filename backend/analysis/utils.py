import json
from typing import Dict, Any

def extract_json(text: str) -> Dict[str, Any]:
    """
    Extracts and parses a JSON object from a string.
    Finds the first '{' and the last '}' and parses the substring,
    handling any markdown code blocks.
    """
    cleaned = text.strip()
    
    # Check for markdown code blocks and extract contents
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    
    if start_idx == -1 or end_idx == -1 or end_idx < start_idx:
        raise ValueError(f"No valid JSON object found in the text. Full text: {text[:200]}...")

    json_str = cleaned[start_idx : end_idx + 1]
    return json.loads(json_str)
