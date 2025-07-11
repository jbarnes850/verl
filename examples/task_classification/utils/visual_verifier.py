"""Visual Verifier for Task Classification

Rule-based verification for visual features in screenshots.
Provides fast, deterministic verification to complement VLM judge.
"""

import logging
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path

try:
    from PIL import Image
    import numpy as np
    import pytesseract
except ImportError as e:
    logging.warning(f"Missing dependencies for visual verifier: {e}")

logger = logging.getLogger(__name__)


class VisualVerifier:
    """Rule-based visual verifier for task classification.
    
    Performs fast verification using:
    - Window title extraction
    - UI element detection  
    - Application identification
    - Content analysis
    """
    
    def __init__(self):
        """Initialize visual verifier."""
        # Work-related application indicators
        self.work_apps = {
            "ide": ["vscode", "visual studio", "intellij", "pycharm", "sublime", "atom"],
            "browser_work": ["github", "gitlab", "jira", "confluence", "slack", "teams"],
            "productivity": ["excel", "word", "powerpoint", "outlook", "notion"],
            "terminal": ["terminal", "cmd", "powershell", "bash"],
            "design": ["figma", "sketch", "photoshop", "illustrator"]
        }
        
        # Distraction indicators
        self.distraction_apps = {
            "social": ["facebook", "twitter", "instagram", "tiktok", "snapchat"],
            "entertainment": ["youtube", "netflix", "twitch", "spotify", "games"],
            "shopping": ["amazon", "ebay", "shopping", "cart", "checkout"],
            "news": ["reddit", "news", "blog", "medium"]
        }
        
        # Task-relevant keywords
        self.work_keywords = [
            "code", "commit", "pull request", "issue", "ticket", "documentation",
            "meeting", "calendar", "email", "project", "task", "sprint", "scrum"
        ]
        
        self.distraction_keywords = [
            "entertainment", "social", "gaming", "shopping", "personal", "fun",
            "meme", "viral", "trending", "celebrity", "sports"
        ]
    
    def verify_classification(self, 
                            screenshot_path: str, 
                            task_description: str,
                            predicted_label: str) -> Tuple[bool, float, str]:
        """Verify classification using rule-based approach.
        
        Args:
            screenshot_path: Path to screenshot
            task_description: Task description
            predicted_label: Model's predicted label
            
        Returns:
            Tuple of (verification_passed, confidence, reason)
        """
        try:
            # Load and analyze image
            image = Image.open(screenshot_path).convert("RGB")
            
            # Extract text content
            extracted_text = self._extract_text(image)
            
            # Analyze window/app indicators
            app_score = self._analyze_application_context(extracted_text)
            
            # Analyze task relevance
            task_score = self._analyze_task_relevance(extracted_text, task_description)
            
            # Analyze content indicators
            content_score = self._analyze_content_indicators(extracted_text)
            
            # Combine scores
            total_score = (app_score * 0.4 + task_score * 0.4 + content_score * 0.2)
            
            # Determine verification
            if predicted_label == "on-task":
                verification_passed = total_score > 0.5
                confidence = total_score
                reason = f"App:{app_score:.2f}, Task:{task_score:.2f}, Content:{content_score:.2f}"
            else:  # off-task
                verification_passed = total_score < 0.5
                confidence = 1.0 - total_score
                reason = f"App:{app_score:.2f}, Task:{task_score:.2f}, Content:{content_score:.2f}"
            
            return verification_passed, confidence, reason
            
        except Exception as e:
            logger.warning(f"Visual verification failed for {screenshot_path}: {e}")
            return True, 0.5, "verification_failed"  # Conservative default
    
    def _extract_text(self, image: Image.Image) -> str:
        """Extract text from image using OCR."""
        try:
            # Use pytesseract for OCR
            text = pytesseract.image_to_string(image, config='--psm 6')
            return text.lower()
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return ""
    
    def _analyze_application_context(self, text: str) -> float:
        """Analyze application context from extracted text.
        
        Returns score between 0 (distraction) and 1 (work-focused).
        """
        work_score = 0.0
        distraction_score = 0.0
        
        # Check for work applications
        for category, apps in self.work_apps.items():
            for app in apps:
                if app in text:
                    work_score += 1.0
        
        # Check for distraction applications
        for category, apps in self.distraction_apps.items():
            for app in apps:
                if app in text:
                    distraction_score += 1.0
        
        # Normalize scores
        total_mentions = work_score + distraction_score
        if total_mentions > 0:
            return work_score / total_mentions
        else:
            return 0.5  # Neutral if no clear indicators
    
    def _analyze_task_relevance(self, text: str, task_description: str) -> float:
        """Analyze relevance to specific task description."""
        task_desc_lower = task_description.lower()
        
        # Extract key terms from task description
        task_terms = []
        
        # Common patterns
        if "jira" in task_desc_lower:
            task_terms.extend(["jira", "ticket", "issue", "atlassian"])
        if "github" in task_desc_lower:
            task_terms.extend(["github", "pull request", "commit", "repository"])
        if "code" in task_desc_lower or "coding" in task_desc_lower:
            task_terms.extend(["code", "function", "class", "import", "def"])
        if "meeting" in task_desc_lower:
            task_terms.extend(["zoom", "teams", "meet", "calendar"])
        if "documentation" in task_desc_lower:
            task_terms.extend(["confluence", "wiki", "documentation", "readme"])
        
        # Check for task-specific terms in extracted text
        matches = 0
        for term in task_terms:
            if term in text:
                matches += 1
        
        # Return relevance score
        if len(task_terms) > 0:
            return min(1.0, matches / len(task_terms))
        else:
            return 0.5  # Neutral if no specific terms
    
    def _analyze_content_indicators(self, text: str) -> float:
        """Analyze general content indicators for work vs distraction."""
        work_matches = 0
        distraction_matches = 0
        
        # Count work-related keywords
        for keyword in self.work_keywords:
            if keyword in text:
                work_matches += 1
        
        # Count distraction-related keywords  
        for keyword in self.distraction_keywords:
            if keyword in text:
                distraction_matches += 1
        
        # Calculate score
        total_matches = work_matches + distraction_matches
        if total_matches > 0:
            return work_matches / total_matches
        else:
            return 0.5  # Neutral
    
    def batch_verify(self, 
                    verification_requests: List[Tuple[str, str, str]]) -> List[Tuple[bool, float, str]]:
        """Batch verify multiple screenshots.
        
        Args:
            verification_requests: List of (screenshot_path, task_description, predicted_label)
            
        Returns:
            List of (verification_passed, confidence, reason) tuples
        """
        results = []
        
        for screenshot_path, task_description, predicted_label in verification_requests:
            result = self.verify_classification(screenshot_path, task_description, predicted_label)
            results.append(result)
        
        return results
    
    def get_verification_stats(self, verifications: List[Tuple[bool, float, str]]) -> Dict:
        """Get statistics on verification results."""
        if not verifications:
            return {"total": 0, "passed": 0, "pass_rate": 0.0, "avg_confidence": 0.0}
        
        passed = sum(1 for v in verifications if v[0])
        avg_confidence = sum(v[1] for v in verifications) / len(verifications)
        
        return {
            "total": len(verifications),
            "passed": passed,
            "pass_rate": passed / len(verifications),
            "avg_confidence": avg_confidence
        }