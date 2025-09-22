"""
ML-based reviewer recommendation system
"""

import re
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
import structlog

logger = structlog.get_logger()


class ReviewerRecommender:
    """ML-based system for recommending code reviewers"""
    
    def __init__(self):
        self.model = None
        self.model_path = Path(settings.ml_model_cache_dir) / "reviewer_model.pkl"
        self.embeddings_cache = {}
        
        # Load sentence transformer model
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.error("Failed to load sentence transformer model", error=str(e))
            self.sentence_model = None
    
    def _extract_file_paths(self, pr_data: Dict) -> List[str]:
        """Extract file paths from PR data"""
        try:
            files = pr_data.get('files', [])
            return [file.get('filename', '') for file in files if file.get('filename')]
        except Exception:
            return []
    
    def _extract_code_changes(self, pr_data: Dict) -> str:
        """Extract code changes summary from PR"""
        try:
            files = pr_data.get('files', [])
            changes = []
            
            for file in files:
                filename = file.get('filename', '')
                additions = file.get('additions', 0)
                deletions = file.get('deletions', 0)
                changes.append(f"{filename}: +{additions}/-{deletions}")
            
            return " ".join(changes)
        except Exception:
            return ""
    
    def _analyze_code_ownership(self, contributors: List[Dict], file_paths: List[str]) -> Dict[str, float]:
        """Analyze code ownership based on file paths and contributor history"""
        ownership_scores = defaultdict(float)
        
        for contributor in contributors:
            username = contributor.get('login', '')
            contributions = contributor.get('contributions', 0)
            
            # Base score from contribution count
            base_score = min(contributions / 100, 1.0)  # Normalize to 0-1
            
            # Check if contributor has worked on similar files
            # This is a simplified heuristic - in practice, you'd analyze commit history
            file_expertise = 0.5  # Placeholder - would analyze actual file history
            
            ownership_scores[username] = base_score * (1 + file_expertise)
        
        return dict(ownership_scores)
    
    def _analyze_expertise_areas(self, contributors: List[Dict], pr_data: Dict) -> Dict[str, float]:
        """Analyze expertise areas based on PR content"""
        if not self.sentence_model:
            return {}
        
        try:
            # Extract PR content
            title = pr_data.get('title', '')
            body = pr_data.get('body', '')
            file_paths = self._extract_file_paths(pr_data)
            
            # Combine all text
            pr_text = f"{title} {body} {' '.join(file_paths)}"
            
            # Generate embedding for PR
            pr_embedding = self.sentence_model.encode([pr_text])[0]
            
            # For each contributor, calculate similarity
            # In practice, you'd have pre-computed contributor profiles
            expertise_scores = {}
            
            for contributor in contributors:
                username = contributor.get('login', '')
                
                # Placeholder: would use actual contributor profile embeddings
                # For now, use a simple heuristic based on contribution count
                contributions = contributor.get('contributions', 0)
                expertise_scores[username] = min(contributions / 50, 1.0)
            
            return expertise_scores
            
        except Exception as e:
            logger.error("Failed to analyze expertise areas", error=str(e))
            return {}
    
    def _check_codeowners(self, file_paths: List[str], codeowners_content: str) -> Dict[str, float]:
        """Check CODEOWNERS file for relevant reviewers"""
        if not codeowners_content:
            return {}
        
        codeowners_scores = defaultdict(float)
        
        try:
            lines = codeowners_content.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse CODEOWNERS line: pattern owners...
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                pattern = parts[0]
                owners = parts[1:]
                
                # Check if any file matches this pattern
                for file_path in file_paths:
                    if self._matches_pattern(pattern, file_path):
                        for owner in owners:
                            # Remove @ prefix if present
                            owner = owner.lstrip('@')
                            codeowners_scores[owner] += 1.0
            
            # Normalize scores
            if codeowners_scores:
                max_score = max(codeowners_scores.values())
                codeowners_scores = {
                    owner: score / max_score 
                    for owner, score in codeowners_scores.items()
                }
            
            return dict(codeowners_scores)
            
        except Exception as e:
            logger.error("Failed to parse CODEOWNERS", error=str(e))
            return {}
    
    def _matches_pattern(self, pattern: str, file_path: str) -> bool:
        """Check if file path matches CODEOWNERS pattern"""
        try:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace('*', '.*').replace('?', '.')
            if not regex_pattern.startswith('^'):
                regex_pattern = '^' + regex_pattern
            if not regex_pattern.endswith('$'):
                regex_pattern = regex_pattern + '$'
            
            return bool(re.match(regex_pattern, file_path))
        except Exception:
            return False
    
    def _calculate_availability_score(self, username: str, recent_prs: List[Dict]) -> float:
        """Calculate availability score based on recent PR activity"""
        try:
            # Count PRs assigned to this user in the last 30 days
            recent_count = 0
            for pr in recent_prs:
                assignees = pr.get('assignees', [])
                if any(assignee.get('login') == username for assignee in assignees):
                    recent_count += 1
            
            # Lower score for users with many recent assignments
            availability = max(0, 1.0 - (recent_count / 10))  # Normalize to 0-1
            return availability
            
        except Exception:
            return 0.5  # Default neutral score
    
    def recommend_reviewers(
        self,
        pr_data: Dict,
        contributors: List[Dict],
        codeowners_content: str = "",
        recent_prs: List[Dict] = None,
        max_reviewers: int = 3
    ) -> List[Tuple[str, float, str]]:
        """Recommend reviewers for a pull request"""
        
        if not contributors:
            return []
        
        recent_prs = recent_prs or []
        file_paths = self._extract_file_paths(pr_data)
        
        # Calculate different types of scores
        ownership_scores = self._analyze_code_ownership(contributors, file_paths)
        expertise_scores = self._analyze_expertise_areas(contributors, pr_data)
        codeowners_scores = self._check_codeowners(file_paths, codeowners_content)
        
        # Combine scores
        final_scores = {}
        
        for contributor in contributors:
            username = contributor.get('login', '')
            
            # Weighted combination of different scores
            ownership = ownership_scores.get(username, 0.0)
            expertise = expertise_scores.get(username, 0.0)
            codeowners = codeowners_scores.get(username, 0.0)
            availability = self._calculate_availability_score(username, recent_prs)
            
            # Weighted score (can be tuned)
            final_score = (
                ownership * 0.3 +
                expertise * 0.3 +
                codeowners * 0.3 +
                availability * 0.1
            )
            
            # Determine reason for recommendation
            reasons = []
            if codeowners > 0:
                reasons.append("CODEOWNERS")
            if ownership > 0.5:
                reasons.append("code_ownership")
            if expertise > 0.5:
                reasons.append("expertise")
            if availability > 0.7:
                reasons.append("availability")
            
            reason = ", ".join(reasons) if reasons else "general"
            
            final_scores[username] = (final_score, reason)
        
        # Sort by score and return top recommendations
        sorted_reviewers = sorted(
            final_scores.items(),
            key=lambda x: x[1][0],
            reverse=True
        )
        
        recommendations = []
        for username, (score, reason) in sorted_reviewers[:max_reviewers]:
            if score > 0.1:  # Minimum threshold
                recommendations.append((username, score, reason))
        
        logger.info(
            "Generated reviewer recommendations",
            pr_number=pr_data.get('number'),
            recommendations_count=len(recommendations)
        )
        
        return recommendations
    
    def get_reviewer_profile(self, username: str, pr_history: List[Dict]) -> Dict[str, any]:
        """Get reviewer profile and statistics"""
        try:
            # Analyze PR review history
            reviewed_prs = [pr for pr in pr_history if username in pr.get('reviewers', [])]
            
            # Calculate statistics
            total_reviews = len(reviewed_prs)
            avg_review_time = 0  # Would calculate from actual review timestamps
            
            # Analyze review patterns
            review_types = defaultdict(int)
            for pr in reviewed_prs:
                # Would analyze actual review content
                review_types['approvals'] += 1
            
            return {
                'username': username,
                'total_reviews': total_reviews,
                'avg_review_time_hours': avg_review_time,
                'review_patterns': dict(review_types),
                'recent_activity': len([pr for pr in reviewed_prs if pr.get('recent', False)])
            }
            
        except Exception as e:
            logger.error("Failed to get reviewer profile", username=username, error=str(e))
            return {'username': username, 'error': str(e)}


class ReviewerService:
    """Service for managing reviewer recommendations"""
    
    def __init__(self):
        self.recommender = ReviewerRecommender()
    
    def recommend_for_pr(
        self,
        pr_data: Dict,
        contributors: List[Dict],
        codeowners_content: str = "",
        recent_prs: List[Dict] = None,
        max_reviewers: int = 3
    ) -> List[Tuple[str, float, str]]:
        """Recommend reviewers for a pull request"""
        return self.recommender.recommend_reviewers(
            pr_data, contributors, codeowners_content, recent_prs, max_reviewers
        )
    
    def get_reviewer_stats(self, username: str, pr_history: List[Dict]) -> Dict[str, any]:
        """Get reviewer statistics and profile"""
        return self.recommender.get_reviewer_profile(username, pr_history)
    
    def analyze_team_review_patterns(self, team_members: List[str], pr_history: List[Dict]) -> Dict[str, any]:
        """Analyze review patterns for a team"""
        try:
            team_stats = {}
            
            for member in team_members:
                team_stats[member] = self.get_reviewer_stats(member, pr_history)
            
            # Calculate team-level statistics
            total_reviews = sum(stats.get('total_reviews', 0) for stats in team_stats.values())
            active_reviewers = len([m for m in team_members if team_stats[m].get('total_reviews', 0) > 0])
            
            return {
                'team_members': team_stats,
                'total_reviews': total_reviews,
                'active_reviewers': active_reviewers,
                'review_distribution': self._calculate_review_distribution(team_stats)
            }
            
        except Exception as e:
            logger.error("Failed to analyze team patterns", error=str(e))
            return {'error': str(e)}
    
    def _calculate_review_distribution(self, team_stats: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate how reviews are distributed across team members"""
        try:
            total_reviews = sum(stats.get('total_reviews', 0) for stats in team_stats.values())
            
            if total_reviews == 0:
                return {}
            
            distribution = {}
            for member, stats in team_stats.items():
                reviews = stats.get('total_reviews', 0)
                distribution[member] = reviews / total_reviews
            
            return distribution
            
        except Exception:
            return {}


# Global reviewer service instance
reviewer_service = ReviewerService()
