"""Prompt templates for PR and Issue generation."""

from typing import List, Dict, Any

PR_GENERATION_PROMPT = """You are an assistant that writes high-quality GitHub pull request descriptions.

Context (similar PRs/issues):
{retrieved_texts}

Diff summary:
{diff_summary}

Commit message:
{commit_msg}

Generate:
1. A concise PR title (< 80 chars)
2. A structured body:
   - Summary
   - Motivation
   - Files changed
   - Related issues
   - Risks / Notes

Format your response as:
TITLE: [title here]

BODY:
[body here with sections]
"""

ISSUE_GENERATION_PROMPT = """You are an assistant that writes clear GitHub issue reports.

Context (similar issues):
{retrieved_texts}

Problem description or diff:
{input_text}

Predicted labels:
{predicted_labels}

Generate a concise issue title and detailed body with:
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (if relevant)

Format your response as:
TITLE: [title here]

BODY:
[body here with sections]
"""


def format_retrieved_context(retrieved_items: List[Dict[str, Any]], max_items: int = 5) -> str:
    """
    Format retrieved items into context text.
    
    Args:
        retrieved_items: List of retrieved records
        max_items: Maximum number of items to include
        
    Returns:
        Formatted context string
    """
    if not retrieved_items:
        return "No similar PRs/issues found."
    
    context_parts = []
    for item in retrieved_items[:max_items]:
        title = item.get("title", "No title")
        text = item.get("text_preview", "")[:500]  # Truncate long texts
        score = item.get("score", 0.0)
        
        context_parts.append(
            f"- [{title}] (similarity: {score:.3f})\n"
            f"  {text}\n"
        )
    
    return "\n".join(context_parts)


def format_pr_prompt(
    diff_text: str,
    commit_msg: str,
    retrieved_items: List[Dict[str, Any]]
) -> str:
    """
    Format PR generation prompt.
    
    Args:
        diff_text: Code diff text
        commit_msg: Commit message
        retrieved_items: Retrieved similar PRs/issues
        
    Returns:
        Formatted prompt string
    """
    retrieved_context = format_retrieved_context(retrieved_items)
    
    return PR_GENERATION_PROMPT.format(
        retrieved_texts=retrieved_context,
        diff_summary=diff_text[:2000] if diff_text else "No diff provided",
        commit_msg=commit_msg or "No commit message"
    )


def format_issue_prompt(
    input_text: str,
    retrieved_items: List[Dict[str, Any]],
    predicted_labels: List[str]
) -> str:
    """
    Format issue generation prompt.
    
    Args:
        input_text: Problem description or diff
        retrieved_items: Retrieved similar issues
        predicted_labels: Predicted labels from labeling model
        
    Returns:
        Formatted prompt string
    """
    retrieved_context = format_retrieved_context(retrieved_items)
    labels_str = ", ".join(predicted_labels) if predicted_labels else "None"
    
    return ISSUE_GENERATION_PROMPT.format(
        retrieved_texts=retrieved_context,
        input_text=input_text[:2000] if input_text else "No description provided",
        predicted_labels=labels_str
    )
