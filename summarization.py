"""
AI-powered summarization for issues and pull requests
"""

import re
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

from app.config import settings
import structlog

logger = structlog.get_logger()


class SummarizationService:
    """AI-powered summarization service for GitHub content"""
    
    def __init__(self):
        self.summarizer = None
        self.sentence_model = None
        self.model_path = Path(settings.ml_model_cache_dir) / "summarization_model"
        
        # Initialize models
        self._load_models()
    
    def _load_models(self):
        """Load summarization models"""
        try:
            # Use a lightweight summarization model
            model_name = "facebook/bart-large-cnn"
            
            # Check if we can use GPU
            device = 0 if torch.cuda.is_available() else -1
            
            self.summarizer = pipeline(
                "summarization",
                model=model_name,
                device=device,
                max_length=150,
                min_length=30,
                do_sample=False
            )
            
            # Load sentence transformer for similarity
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Loaded summarization models", model=model_name, device=device)
            
        except Exception as e:
            logger.error("Failed to load summarization models", error=str(e))
            self.summarizer = None
            self.sentence_model = None
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for summarization"""
        if not text:
            return ""
        
        # Remove markdown formatting
        text = re.sub(r'```.*?```', '[CODE_BLOCK]', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', '[CODE]', text)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'#{1,6}\s+', '', text)         # Headers
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # Links
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from text using sentence similarity"""
        if not self.sentence_model or not text:
            return []
        
        try:
            # Split into sentences
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) <= 3:
                return sentences
            
            # Generate embeddings
            embeddings = self.sentence_model.encode(sentences)
            
            # Calculate similarity matrix
            similarity_matrix = torch.cosine_similarity(
                embeddings.unsqueeze(1), 
                embeddings.unsqueeze(0), 
                dim=2
            )
            
            # Find most representative sentences
            key_sentences = []
            used_indices = set()
            
            for _ in range(min(3, len(sentences))):
                best_score = -1
                best_idx = -1
                
                for i, sentence in enumerate(sentences):
                    if i in used_indices:
                        continue
                    
                    # Calculate average similarity to other sentences
                    similarities = similarity_matrix[i]
                    avg_similarity = similarities.mean().item()
                    
                    if avg_similarity > best_score:
                        best_score = avg_similarity
                        best_idx = i
                
                if best_idx >= 0:
                    key_sentences.append(sentences[best_idx])
                    used_indices.add(best_idx)
            
            return key_sentences
            
        except Exception as e:
            logger.error("Failed to extract key points", error=str(e))
            return []
    
    def summarize_issue(self, title: str, body: str = "") -> Dict[str, str]:
        """Summarize a GitHub issue"""
        try:
            # Combine title and body
            full_text = f"{title}\n\n{body}".strip()
            
            if not full_text:
                return {"summary": "No content to summarize", "key_points": []}
            
            # Preprocess
            processed_text = self._preprocess_text(full_text)
            
            if len(processed_text.split()) < 10:
                return {
                    "summary": processed_text,
                    "key_points": [processed_text]
                }
            
            # Generate summary
            if self.summarizer:
                try:
                    # Truncate if too long
                    max_length = 1024
                    if len(processed_text) > max_length:
                        processed_text = processed_text[:max_length]
                    
                    summary_result = self.summarizer(
                        processed_text,
                        max_length=100,
                        min_length=20,
                        do_sample=False
                    )
                    
                    summary = summary_result[0]['summary_text']
                except Exception as e:
                    logger.warning("Summarization failed, using fallback", error=str(e))
                    summary = self._fallback_summary(processed_text)
            else:
                summary = self._fallback_summary(processed_text)
            
            # Extract key points
            key_points = self._extract_key_points(processed_text)
            
            return {
                "summary": summary,
                "key_points": key_points[:3]  # Limit to top 3
            }
            
        except Exception as e:
            logger.error("Failed to summarize issue", error=str(e))
            return {
                "summary": "Failed to generate summary",
                "key_points": []
            }
    
    def summarize_pull_request(
        self, 
        title: str, 
        body: str = "", 
        files_changed: List[Dict] = None
    ) -> Dict[str, str]:
        """Summarize a GitHub pull request"""
        try:
            # Combine all content
            content_parts = [title]
            
            if body:
                content_parts.append(body)
            
            if files_changed:
                file_summary = self._summarize_file_changes(files_changed)
                if file_summary:
                    content_parts.append(f"Files changed: {file_summary}")
            
            full_text = "\n\n".join(content_parts)
            
            if not full_text:
                return {"summary": "No content to summarize", "key_points": []}
            
            # Preprocess
            processed_text = self._preprocess_text(full_text)
            
            if len(processed_text.split()) < 10:
                return {
                    "summary": processed_text,
                    "key_points": [processed_text]
                }
            
            # Generate summary
            if self.summarizer:
                try:
                    # Truncate if too long
                    max_length = 1024
                    if len(processed_text) > max_length:
                        processed_text = processed_text[:max_length]
                    
                    summary_result = self.summarizer(
                        processed_text,
                        max_length=120,
                        min_length=30,
                        do_sample=False
                    )
                    
                    summary = summary_result[0]['summary_text']
                except Exception as e:
                    logger.warning("Summarization failed, using fallback", error=str(e))
                    summary = self._fallback_summary(processed_text)
            else:
                summary = self._fallback_summary(processed_text)
            
            # Extract key points
            key_points = self._extract_key_points(processed_text)
            
            return {
                "summary": summary,
                "key_points": key_points[:3]  # Limit to top 3
            }
            
        except Exception as e:
            logger.error("Failed to summarize pull request", error=str(e))
            return {
                "summary": "Failed to generate summary",
                "key_points": []
            }
    
    def _summarize_file_changes(self, files_changed: List[Dict]) -> str:
        """Summarize file changes in a PR"""
        try:
            if not files_changed:
                return ""
            
            # Group files by type
            file_types = defaultdict(list)
            
            for file_info in files_changed:
                filename = file_info.get('filename', '')
                additions = file_info.get('additions', 0)
                deletions = file_info.get('deletions', 0)
                
                # Determine file type
                if filename.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c')):
                    file_type = 'code'
                elif filename.endswith(('.md', '.txt', '.rst')):
                    file_type = 'docs'
                elif filename.endswith(('.yml', '.yaml', '.json', '.xml')):
                    file_type = 'config'
                elif filename.endswith(('.test', '.spec')):
                    file_type = 'tests'
                else:
                    file_type = 'other'
                
                file_types[file_type].append({
                    'name': filename,
                    'additions': additions,
                    'deletions': deletions
                })
            
            # Create summary
            summary_parts = []
            
            for file_type, files in file_types.items():
                if not files:
                    continue
                
                total_additions = sum(f['additions'] for f in files)
                total_deletions = sum(f['deletions'] for f in files)
                
                summary_parts.append(
                    f"{file_type}: {len(files)} files (+{total_additions}/-{total_deletions})"
                )
            
            return ", ".join(summary_parts)
            
        except Exception as e:
            logger.error("Failed to summarize file changes", error=str(e))
            return ""
    
    def _fallback_summary(self, text: str) -> str:
        """Fallback summarization when ML model fails"""
        try:
            # Simple extractive summarization
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if not sentences:
                return text
            
            # Take first few sentences
            if len(sentences) <= 2:
                return " ".join(sentences)
            else:
                return " ".join(sentences[:2]) + "..."
                
        except Exception:
            return text[:200] + "..." if len(text) > 200 else text
    
    def generate_pr_description(
        self,
        title: str,
        body: str = "",
        files_changed: List[Dict] = None,
        template: str = None
    ) -> str:
        """Generate a comprehensive PR description"""
        try:
            # Get summary
            summary_data = self.summarize_pull_request(title, body, files_changed)
            summary = summary_data['summary']
            key_points = summary_data['key_points']
            
            # Use template if provided
            if template:
                return template.format(
                    summary=summary,
                    key_points="\n".join(f"- {point}" for point in key_points),
                    title=title
                )
            
            # Default template
            description_parts = [f"## Summary\n{summary}"]
            
            if key_points:
                description_parts.append("## Key Changes")
                for point in key_points:
                    description_parts.append(f"- {point}")
            
            if files_changed:
                file_summary = self._summarize_file_changes(files_changed)
                if file_summary:
                    description_parts.append(f"## Files Changed\n{file_summary}")
            
            return "\n\n".join(description_parts)
            
        except Exception as e:
            logger.error("Failed to generate PR description", error=str(e))
            return body or title


# Global summarization service instance
summarization_service = SummarizationService()
