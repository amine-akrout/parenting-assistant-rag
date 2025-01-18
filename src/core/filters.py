# src/core/filters.py
"""
This module initializes the LLM Guard input and output scanners.
It is loaded once when the API starts.
"""

from llm_guard.input_scanners import BanTopics, Language, PromptInjection, Toxicity
from llm_guard.output_scanners import LanguageSame, Relevance, Sensitive

# Define Input Scanners
INPUT_SCANNERS = [
    Toxicity(),
    PromptInjection(),
    Language(valid_languages=["en"]),
    BanTopics(
        topics=[
            "politics",
            "religion",
            "drugs",
            "weapons",
            "illegal activities",
            "sexual content",
            "discrimination",
        ]
    ),
]

# Define Output Scanners
OUTPUT_SCANNERS = [
    LanguageSame(),
    Relevance(),
    Sensitive(),
]


# Function to get scanners
def get_input_scanners():
    """
    Get the input scanners.
    """
    return INPUT_SCANNERS


def get_output_scanners():
    """
    Get the output scanners.
    """
    return OUTPUT_SCANNERS
