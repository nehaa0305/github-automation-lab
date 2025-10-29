"""Text generation with pluggable backends for RAG."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)


class Generator:
    """Base generator class with pluggable backends."""
    
    def __init__(self, model_name: str = "codet5-base", backend_type: str = "codet5"):
        """
        Initialize generator.
        
        Args:
            model_name: Name of the model to use
            backend_type: Type of backend ("codet5", "codebert", "gpt", "simple")
        """
        self.model_name = model_name
        self.backend_type = backend_type
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the generation model."""
        if self.backend_type == "codet5":
            try:
                from transformers import T5ForConditionalGeneration, T5Tokenizer
                logger.info(f"Loading CodeT5 model: {self.model_name}")
                self.tokenizer = T5Tokenizer.from_pretrained(f"Salesforce/{self.model_name}")
                self.model = T5ForConditionalGeneration.from_pretrained(f"Salesforce/{self.model_name}")
                logger.info("CodeT5 model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load CodeT5: {e}. Using simple generator.")
                self.backend_type = "simple"
        
        elif self.backend_type == "codebert":
            try:
                from transformers import GPT2LMHeadModel, GPT2Tokenizer
                logger.info(f"Loading CodeBERT-style model: {self.model_name}")
                self.tokenizer = GPT2Tokenizer.from_pretrained(f"microsoft/{self.model_name}")
                self.model = GPT2LMHeadModel.from_pretrained(f"microsoft/{self.model_name}")
                logger.info("CodeBERT-style model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load CodeBERT: {e}. Using simple generator.")
                self.backend_type = "simple"
    
    def generate(self, prompt: str, max_length: int = 512) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated text
        """
        if self.backend_type == "simple":
            return self._simple_generate(prompt)
        
        try:
            if self.backend_type == "codet5":
                return self._codet5_generate(prompt, max_length)
            elif self.backend_type == "codebert":
                return self._codebert_generate(prompt, max_length)
        except Exception as e:
            logger.error(f"Generation failed: {e}. Falling back to simple generator.")
            return self._simple_generate(prompt)
    
    def _codet5_generate(self, prompt: str, max_length: int) -> str:
        """Generate using CodeT5."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def _codebert_generate(self, prompt: str, max_length: int) -> str:
        """Generate using CodeBERT-style model."""
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(
            inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated
    
    def _simple_generate(self, prompt: str) -> str:
        """
        Simple template-based generation (fallback).
        Extracts key information from prompt and formats it.
        """
        # Extract diff summary
        diff_match = re.search(r'Diff summary:\s*(.+?)(?=Commit message:|$)', prompt, re.DOTALL)
        diff_text = diff_match.group(1).strip() if diff_match else ""
        
        # Extract commit message
        commit_match = re.search(r'Commit message:\s*(.+?)(?=Generate:|$)', prompt, re.DOTALL)
        commit_msg = commit_match.group(1).strip() if commit_match else ""
        
        # Generate simple title from commit or first line of diff
        if commit_msg:
            title = commit_msg.split('\n')[0][:80]
        elif diff_text:
            title = diff_text.split('\n')[0][:80]
        else:
            title = "Code changes"
        
        # Generate simple body
        body = f"""## Summary
{commit_msg if commit_msg else 'Code changes and improvements'}

## Changes
{self._extract_file_changes(diff_text) if diff_text else 'Files modified'}

## Notes
Generated automatically using RAG system.
"""
        
        return f"TITLE: {title}\n\nBODY:\n{body}"
    
    def _extract_file_changes(self, diff_text: str) -> str:
        """Extract file names from diff."""
        file_pattern = r'(?:---|\+\+\+) a?/?([^\s]+)'
        files = set(re.findall(file_pattern, diff_text))
        if files:
            return "- " + "\n- ".join(list(files)[:10])  # Limit to 10 files
        return "Files modified"


def generate_pr_description(
    diff_text: str,
    commit_msg: str,
    retrieved_prs: List[Dict[str, Any]],
    model_name: str = "codet5-base",
    backend_type: str = "simple"
) -> Dict[str, str]:
    """
    Generate PR description using RAG.
    
    Args:
        diff_text: Code diff text
        commit_msg: Commit message
        retrieved_prs: Retrieved similar PRs/issues
        model_name: Model name for generation
        backend_type: Backend type ("codet5", "codebert", "simple")
        
    Returns:
        Dictionary with 'title' and 'body'
    """
    from .prompts import format_pr_prompt
    from .generator import Generator
    
    logger.info("Generating PR description...")
    
    # Fast path: high-quality deterministic template without heavy models
    if backend_type == "simple":
        title, body = _build_contextual_pr_template(diff_text or "", commit_msg or "", retrieved_prs or [])
    else:
        # Create generator
        generator = Generator(model_name=model_name, backend_type=backend_type)
        # Format prompt
        prompt = format_pr_prompt(diff_text, commit_msg, retrieved_prs)
        # Generate
        generated_text = generator.generate(prompt, max_length=512)
        # Parse generated text
        title, body = parse_generated_text(generated_text)
    
    logger.info(f"Generated PR: {title[:50]}...")
    
    return {
        "title": title,
        "body": body,
        "retrieved_count": len(retrieved_prs)
    }


def generate_issue_summary(
    input_text: str,
    retrieved_issues: List[Dict[str, Any]],
    predicted_labels: List[str],
    model_name: str = "codet5-base",
    backend_type: str = "simple"
) -> Dict[str, str]:
    """
    Generate issue summary using RAG.
    
    Args:
        input_text: Problem description or diff
        retrieved_issues: Retrieved similar issues
        predicted_labels: Predicted labels from labeling model
        model_name: Model name for generation
        backend_type: Backend type
        
    Returns:
        Dictionary with 'title' and 'body'
    """
    from .prompts import format_issue_prompt
    from .generator import Generator
    
    logger.info("Generating issue summary...")
    
    if backend_type == "simple":
        title, body = _build_contextual_issue_template(input_text or "", retrieved_issues or [], predicted_labels or [])
    else:
        # Create generator
        generator = Generator(model_name=model_name, backend_type=backend_type)
        # Format prompt
        prompt = format_issue_prompt(input_text, retrieved_issues, predicted_labels)
        # Generate
        generated_text = generator.generate(prompt, max_length=512)
        # Parse generated text
        title, body = parse_generated_text(generated_text)
    
    logger.info(f"Generated Issue: {title[:50]}...")
    
    return {
        "title": title,
        "body": body,
        "predicted_labels": predicted_labels,
        "retrieved_count": len(retrieved_issues)
    }


def parse_generated_text(generated_text: str) -> tuple:
    """
    Parse generated text into title and body.
    
    Args:
        generated_text: Generated text with TITLE: and BODY: markers
        
    Returns:
        Tuple of (title, body)
    """
    # Extract title
    title_match = re.search(r'TITLE:\s*(.+?)(?=\nBODY:|\n\n|$)', generated_text, re.DOTALL)
    if title_match:
        title = title_match.group(1).strip()
    else:
        # Fallback: use first line
        title = generated_text.split('\n')[0].strip()[:80]
    
    # Extract body
    body_match = re.search(r'BODY:\s*(.+?)$', generated_text, re.DOTALL)
    if body_match:
        body = body_match.group(1).strip()
    else:
        # Fallback: use everything after title
        body = generated_text.replace(f"TITLE: {title}", "").strip()
        if not body:
            body = generated_text.strip()
    
    return title, body


# -----------------------------
# Context-aware deterministic templates
# -----------------------------

def _parse_diff_files(diff_text: str) -> List[Dict[str, Any]]:
    """Parse unified diff to extract per-file changes and quick stats."""
    files: List[Dict[str, Any]] = []
    if not diff_text:
        return files
    current: Optional[Dict[str, Any]] = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            if current:
                files.append(current)
            parts = line.strip().split()
            path_a = parts[2][2:] if len(parts) > 2 else ""
            path_b = parts[3][2:] if len(parts) > 3 else ""
            current = {"file": path_b or path_a, "add": 0, "del": 0, "hunks": []}
        elif line.startswith("@@"):
            if current is not None:
                current["hunks"].append({"lines": []})
        elif line.startswith("+") and not line.startswith("+++ "):
            if current is not None:
                current["add"] += 1
                if current["hunks"]:
                    current["hunks"][-1]["lines"].append(line[1:])
        elif line.startswith("-") and not line.startswith("--- "):
            if current is not None:
                current["del"] += 1
                if current["hunks"]:
                    current["hunks"][-1]["lines"].append(line[1:])
    if current:
        files.append(current)
    return files


def _summarize_files(files: List[Dict[str, Any]], max_files: int = 10) -> str:
    if not files:
        return "(no files detected)"
    lines: List[str] = []
    for f in files[:max_files]:
        lines.append(f"- `{f['file']}` (+{f['add']}/-{f['del']})")
    return "\n".join(lines)


def _top_snippets(files: List[Dict[str, Any]], max_files: int = 3, max_lines: int = 12) -> str:
    if not files:
        return "(no snippets)"
    blocks: List[str] = []
    for f in files[:max_files]:
        collected: List[str] = []
        for h in f.get("hunks", [])[:2]:
            for l in h.get("lines", [])[: max_lines // 2]:
                collected.append(l)
        if collected:
            snippet = "\n".join(collected[:max_lines])
            blocks.append(f"### {f['file']}\n```diff\n{snippet}\n```")
    return "\n\n".join(blocks) if blocks else "(no diff excerpts)"


def _format_retrieved(retrieved: List[Dict[str, Any]], kind: str, max_items: int = 5) -> str:
    if not retrieved:
        return f"(no related {kind})"
    lines: List[str] = []
    for item in retrieved[:max_items]:
        title = item.get("title") or item.get("id") or "Untitled"
        score = item.get("score", 0.0)
        preview = (item.get("text_preview") or "").strip().replace("\n", " ")
        if len(preview) > 180:
            preview = preview[:177] + "..."
        lines.append(f"- {title} (sim {score:.2f}) — {preview}")
    return "\n".join(lines)


def _build_contextual_pr_template(diff_text: str, commit_msg: str, retrieved: List[Dict[str, Any]]) -> tuple:
    files = _parse_diff_files(diff_text)
    title_base = commit_msg.split("\n")[0].strip() if commit_msg else "Update code"
    if len(title_base) > 80:
        title_base = title_base[:77] + "..."
    files_list = _summarize_files(files)
    snippets = _top_snippets(files)
    related = _format_retrieved(retrieved, "PRs/issues")

    body = (
        f"## Summary\n{commit_msg or 'This PR updates repository code based on recent changes.'}\n\n"
        f"## Motivation\nExplain why this change is necessary and how it aligns with project goals.\n\n"
        f"## Implementation Details\n- Key changes across modules\n- Important functions altered\n- Data/model interactions\n\n"
        f"## Files Changed\n{files_list}\n\n"
        f"## Diff Excerpts\n{snippets}\n\n"
        f"## Related Context\n{related}\n\n"
        f"## Tests\n- [ ] Unit tests for modified logic\n- [ ] Integration/regression coverage\n\n"
        f"## Risks & Rollback\n- Potential side effects and mitigation\n- Rollback plan if issues occur\n"
    )
    return title_base, body


def _build_contextual_issue_template(input_text: str, retrieved: List[Dict[str, Any]], labels: List[str]) -> tuple:
    title_base = (input_text.split("\n")[0][:80].strip() if input_text else "Auto-generated issue").rstrip()
    related = _format_retrieved(retrieved, "issues")
    labels_md = ", ".join(labels) if labels else "none"
    body = (
        f"## Summary\n{input_text[:600] if input_text else 'Code or behavior change detected.'}\n\n"
        f"## Steps to Reproduce\n1. ...\n2. ...\n3. ...\n\n"
        f"## Expected vs Actual\n- Expected: ...\n- Actual: ...\n\n"
        f"## Related Context\n{related}\n\n"
        f"## Suggested Fix\nDescribe the suspected root cause and proposed fix.\n\n"
        f"## Labels\n{labels_md}\n"
    )
    return title_base, body
