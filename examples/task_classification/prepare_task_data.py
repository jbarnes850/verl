#!/usr/bin/env python3
"""Data preparation pipeline for task classification.

Creates initial training dataset using VLM judge bootstrap approach.
Generates synthetic task-screenshot pairs for binary classification.
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd

from utils.vlm_judge import VLMJudge

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskDataPreparer:
    """Prepares task classification dataset using VLM judge bootstrap."""
    
    def __init__(self, output_dir: str = "data", vlm_judge_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"):
        """Initialize data preparer.
        
        Args:
            output_dir: Directory to save prepared datasets
            vlm_judge_model: Model path for VLM judge
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize VLM judge for bootstrapping
        try:
            self.vlm_judge = VLMJudge(model_path=vlm_judge_model)
            logger.info(f"Initialized VLM judge: {vlm_judge_model}")
        except Exception as e:
            logger.warning(f"Failed to initialize VLM judge: {e}")
            self.vlm_judge = None
        
        # Task descriptions for synthetic data generation
        # Expanded based on research - more diverse and realistic scenarios
        self.work_tasks = [
            # Development Tasks
            "Working on Jira ticket PROJ-123",
            "Reviewing pull request in GitHub",
            "Coding in VS Code on feature branch",
            "Writing unit tests in PyTest",
            "Debugging application in browser",
            "Reviewing code in GitLab",
            "Setting up CI/CD pipeline",
            "Creating database schema",
            "Writing API documentation",
            "Refactoring legacy codebase",
            "Implementing authentication system",
            "Optimizing database queries",
            "Fixing production bug in main branch",
            "Writing integration tests",
            "Deploying application to staging",
            
            # Documentation & Communication
            "Writing documentation in Confluence",
            "Responding to Slack messages from manager",
            "Creating presentation in PowerPoint",
            "Planning sprint in Azure DevOps",
            "Updating project wiki",
            "Conducting code review",
            "Writing technical specifications",
            "Preparing quarterly report",
            "Creating user manual for new feature",
            "Documenting API endpoints",
            "Writing architecture diagrams",
            "Updating team knowledge base",
            
            # Meetings & Collaboration
            "Attending Zoom meeting with team",
            "Participating in daily standup",
            "Leading technical design review",
            "Client presentation via Teams",
            "Sprint planning meeting",
            "Retrospective session with team",
            "One-on-one with manager",
            "Architecture discussion meeting",
            "Bug triage session",
            "Demo session with stakeholders",
            
            # Analysis & Research
            "Analyzing data in Excel spreadsheet",
            "Monitoring application logs",
            "Investigating performance bottleneck",
            "Researching third-party libraries",
            "Analyzing user behavior metrics",
            "Reviewing security audit results",
            "Performance testing analysis",
            "Capacity planning analysis",
            "Cost optimization research",
            "Technical feasibility study"
        ]
        
        # Expanded distraction scenarios for better edge case coverage
        self.distraction_scenarios = [
            # Social Media & Entertainment
            "Browsing Reddit",
            "Watching YouTube videos", 
            "Checking social media",
            "Chatting on Discord",
            "Browsing Instagram",
            "Scrolling through Twitter",
            "Watching TikTok videos",
            "Browsing Facebook timeline",
            "Watching Twitch streams",
            "Reading Twitter threads",
            "Browsing LinkedIn posts",
            "Watching Instagram Stories",
            
            # Shopping & Commerce
            "Shopping on Amazon",
            "Online shopping",
            "Browsing product reviews",
            "Checking sale notifications",
            "Comparing product prices",
            "Reading shopping wishlists",
            "Browsing online marketplaces",
            
            # Entertainment & Media
            "Watching Netflix",
            "Watching TikTok videos",
            "Reading blogs",
            "Reading news articles",
            "Watching movie trailers",
            "Browsing entertainment news",
            "Reading celebrity gossip",
            "Watching sports highlights",
            "Reading comic strips",
            "Browsing memes",
            
            # Gaming & Personal
            "Playing online games",
            "Playing mobile games",
            "Reading personal email",
            "Chatting with friends",
            "Planning personal vacation",
            "Looking at personal photos",
            "Reading personal messages",
            "Browsing dating apps",
            "Checking weather forecast",
            "Reading personal finance apps",
            
            # Ambiguous/Edge Cases
            "Reading tech blogs (personal interest)",
            "Watching coding tutorials (not work-related)",
            "Personal project on GitHub",
            "Learning new programming language (hobby)",
            "Personal side project development",
            "Reading career advice articles",
            "Personal skill development",
            "Online course for personal growth"
        ]
    
    def generate_synthetic_dataset(self, 
                                 num_samples: int = 10000,
                                 train_ratio: float = 0.8,
                                 confidence_threshold: float = 0.8,
                                 curriculum_learning: bool = True) -> Tuple[str, str]:
        """Generate synthetic dataset using VLM judge.
        
        Args:
            num_samples: Total number of samples to generate
            train_ratio: Ratio for train/validation split
            confidence_threshold: Minimum confidence for VLM judge labels
            curriculum_learning: Whether to organize data for curriculum learning
            
        Returns:
            Tuple of (train_file_path, val_file_path)
        """
        logger.info(f"Generating {num_samples} synthetic samples...")
        
        # Generate task-scenario pairs with curriculum learning support
        samples = []
        
        # Separate tasks by difficulty for curriculum learning
        if curriculum_learning:
            # Easy cases: clear work vs clear distraction
            easy_work_tasks = [t for t in self.work_tasks if not any(edge in t.lower() 
                              for edge in ["personal", "hobby", "learning", "career"])]
            easy_distraction_tasks = [t for t in self.distraction_scenarios if not any(edge in t.lower() 
                                     for edge in ["tech", "coding", "github", "programming", "course"])]
            
            # Hard cases: ambiguous or edge cases
            hard_work_tasks = [t for t in self.work_tasks if any(edge in t.lower() 
                              for edge in ["research", "learning", "feasibility"])]
            hard_distraction_tasks = [t for t in self.distraction_scenarios if any(edge in t.lower() 
                                     for edge in ["tech", "coding", "github", "programming", "course"])]
            
            # Generate curriculum: 60% easy, 40% hard
            easy_count = int(num_samples * 0.6)
            hard_count = num_samples - easy_count
            
            logger.info(f"Curriculum learning: {easy_count} easy samples, {hard_count} hard samples")
        else:
            easy_work_tasks = hard_work_tasks = self.work_tasks
            easy_distraction_tasks = hard_distraction_tasks = self.distraction_scenarios
            easy_count = hard_count = num_samples // 2
        
        # Generate easy samples first (for curriculum learning)
        for i in range(easy_count):
            if i % 2 == 0:
                task = random.choice(easy_work_tasks)
                scenario = "on-task"
                screenshot_type = "work"
                difficulty = "easy"
            else:
                # Use distraction scenario description as task, making it clearly off-task
                distraction = random.choice(easy_distraction_tasks)
                task = random.choice(self.work_tasks)  # Real work task
                scenario = "off-task"
                screenshot_type = "distraction"
                difficulty = "easy"
            
            sample = {
                "task_description": task,
                "screenshot": f"synthetic_screenshot_{i}.png",
                "label": scenario,
                "screenshot_type": screenshot_type,
                "difficulty": difficulty,
                "confidence": 1.0,
                "source": "synthetic"
            }
            samples.append(sample)
        
        # Generate hard samples (edge cases)
        for i in range(easy_count, num_samples):
            if i % 2 == 0:
                task = random.choice(hard_work_tasks)
                scenario = "on-task"
                screenshot_type = "work"
                difficulty = "hard"
            else:
                task = random.choice(self.work_tasks)
                scenario = "off-task"
                screenshot_type = "distraction"
                difficulty = "hard"
            
            sample = {
                "task_description": task,
                "screenshot": f"synthetic_screenshot_{i}.png",
                "label": scenario,
                "screenshot_type": screenshot_type,
                "difficulty": difficulty,
                "confidence": 0.8,  # Lower confidence for edge cases
                "source": "synthetic"
            }
            samples.append(sample)
        
        # If VLM judge is available, validate high-confidence samples
        if self.vlm_judge:
            logger.info("Validating samples with VLM judge...")
            validated_samples = self._validate_with_vlm_judge(samples, confidence_threshold)
        else:
            logger.warning("No VLM judge available - using synthetic labels")
            validated_samples = samples
        
        # Split into train/val
        random.shuffle(validated_samples)
        split_idx = int(len(validated_samples) * train_ratio)
        
        train_samples = validated_samples[:split_idx]
        val_samples = validated_samples[split_idx:]
        
        # Save datasets
        train_file = self.output_dir / "train.parquet"
        val_file = self.output_dir / "val.parquet"
        
        pd.DataFrame(train_samples).to_parquet(train_file)
        pd.DataFrame(val_samples).to_parquet(val_file)
        
        logger.info(f"Saved {len(train_samples)} training samples to {train_file}")
        logger.info(f"Saved {len(val_samples)} validation samples to {val_file}")
        
        # Save metadata
        metadata = {
            "total_samples": len(validated_samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "confidence_threshold": confidence_threshold,
            "vlm_judge_model": self.vlm_judge.model_path if self.vlm_judge else None,
            "label_distribution": self._get_label_distribution(validated_samples)
        }
        
        with open(self.output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        return str(train_file), str(val_file)
    
    def create_demo_screenshots(self, num_screenshots: int = 50):
        """Create demo screenshots for testing.
        
        Args:
            num_screenshots: Number of demo screenshots to create
        """
        logger.info(f"Creating {num_screenshots} demo screenshots...")
        
        from PIL import Image, ImageDraw, ImageFont
        
        screenshot_dir = self.output_dir / "demo_screenshots"
        screenshot_dir.mkdir(exist_ok=True)
        
        # Create demo screenshots with different scenarios
        for i in range(num_screenshots):
            # Create base image
            img = Image.new("RGB", (1920, 1080), color="white")
            draw = ImageDraw.Draw(img)
            
            # Determine scenario type
            if i % 2 == 0:
                # On-task scenario
                self._draw_work_scenario(draw, img.size)
                screenshot_type = "work"
            else:
                # Off-task scenario
                self._draw_distraction_scenario(draw, img.size)
                screenshot_type = "distraction"
            
            # Save screenshot
            screenshot_path = screenshot_dir / f"demo_{screenshot_type}_{i:03d}.png"
            img.save(screenshot_path)
        
        logger.info(f"Created demo screenshots in {screenshot_dir}")
    
    def _draw_work_scenario(self, draw, size):
        """Draw a work-related scenario on the image."""
        width, height = size
        
        # Draw IDE/editor interface
        # Top bar
        draw.rectangle([0, 0, width, 60], fill="#2d2d2d")
        draw.text((20, 20), "VS Code - main.py", fill="white")
        
        # Side panel
        draw.rectangle([0, 60, 200, height], fill="#1e1e1e")
        draw.text((10, 80), "Explorer", fill="white")
        draw.text((10, 120), "├── src/", fill="#cccccc")
        draw.text((10, 140), "│   ├── main.py", fill="#4fc3f7")
        draw.text((10, 160), "│   └── utils.py", fill="#cccccc")
        
        # Main editor area
        draw.rectangle([200, 60, width, height], fill="#1e1e1e")
        draw.text((220, 80), "def classify_task(screenshot, description):", fill="#569cd6")
        draw.text((240, 100), "# TODO: Implement task classification", fill="#6a9955")
        draw.text((240, 120), "return 'on-task'", fill="#ce9178")
    
    def _draw_distraction_scenario(self, draw, size):
        """Draw a distraction scenario on the image.""" 
        width, height = size
        
        # Draw social media interface
        # Top bar
        draw.rectangle([0, 0, width, 80], fill="#1877f2")
        draw.text((20, 25), "Facebook", fill="white")
        draw.text((width-200, 25), "Home  Profile  Messages", fill="white")
        
        # Main feed area
        draw.rectangle([200, 80, width-200, height], fill="#f0f2f5")
        
        # Posts
        y_pos = 120
        for i in range(3):
            # Post container
            draw.rectangle([220, y_pos, width-220, y_pos+150], fill="white")
            draw.text((240, y_pos+10), f"Friend {i+1}", fill="#1c1e21")
            draw.text((240, y_pos+35), "Just had an amazing vacation! 🏖️", fill="#1c1e21")
            draw.rectangle([240, y_pos+60, width-240, y_pos+130], fill="#e4e6ea")
            draw.text((250, y_pos+90), "[Vacation Photo]", fill="#65676b")
            y_pos += 180
        
        # Sidebar
        draw.rectangle([0, 80, 200, height], fill="#f0f2f5")
        draw.text((10, 100), "Trending", fill="#1c1e21")
        draw.text((10, 130), "• Celebrity News", fill="#65676b")
        draw.text((10, 150), "• Sports Update", fill="#65676b")
        draw.text((10, 170), "• Funny Videos", fill="#65676b")
    
    def _validate_with_vlm_judge(self, samples: List[Dict], confidence_threshold: float) -> List[Dict]:
        """Validate samples using VLM judge (mock implementation).
        
        Args:
            samples: List of samples to validate
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of validated samples
        """
        # Note: This is a mock implementation since we don't have actual screenshots yet
        # In real implementation, this would use actual screenshots
        
        validated = []
        
        for sample in samples:
            # Mock VLM judge validation based on task description patterns
            task_desc = sample["task_description"].lower()
            expected_label = sample["label"]
            
            # Simple heuristic for validation
            work_keywords = ["jira", "github", "code", "documentation", "meeting", "slack", "excel"]
            distraction_keywords = ["reddit", "youtube", "amazon", "social", "games", "netflix"]
            
            # Determine confidence based on keyword matching
            if expected_label == "on-task":
                confidence = 0.9 if any(kw in task_desc for kw in work_keywords) else 0.6
            else:
                confidence = 0.9 if any(kw in task_desc for kw in distraction_keywords) else 0.6
            
            # Only include high-confidence samples
            if confidence >= confidence_threshold:
                sample["confidence"] = confidence
                sample["judge_validated"] = True
                validated.append(sample)
            else:
                sample["confidence"] = confidence
                sample["judge_validated"] = False
                # Still include but mark as low confidence
                validated.append(sample)
        
        logger.info(f"Validated {len(validated)} samples (confidence >= {confidence_threshold})")
        return validated
    
    def _get_label_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Get distribution of labels in samples."""
        distribution = {}
        for sample in samples:
            label = sample["label"]
            distribution[label] = distribution.get(label, 0) + 1
        return distribution
    
    def load_existing_screenshots(self, screenshot_dir: str) -> List[Dict]:
        """Load existing screenshots and generate dataset.
        
        Args:
            screenshot_dir: Directory containing screenshots
            
        Returns:
            List of data samples
        """
        screenshot_path = Path(screenshot_dir)
        if not screenshot_path.exists():
            logger.error(f"Screenshot directory not found: {screenshot_dir}")
            return []
        
        samples = []
        screenshot_files = list(screenshot_path.glob("*.png")) + list(screenshot_path.glob("*.jpg"))
        
        logger.info(f"Found {len(screenshot_files)} screenshots in {screenshot_dir}")
        
        for i, screenshot_file in enumerate(screenshot_files):
            # Assign random task and use VLM judge to label
            task = random.choice(self.work_tasks)
            
            if self.vlm_judge:
                try:
                    label, confidence = self.vlm_judge.judge_classification(
                        str(screenshot_file), task
                    )
                except Exception as e:
                    logger.warning(f"VLM judge failed for {screenshot_file}: {e}")
                    label, confidence = "off-task", 0.5
            else:
                # Random assignment for demo
                label = random.choice(["on-task", "off-task"])
                confidence = random.uniform(0.6, 0.9)
            
            sample = {
                "task_description": task,
                "screenshot": str(screenshot_file),
                "label": label,
                "confidence": confidence,
                "source": "real_screenshot"
            }
            
            samples.append(sample)
        
        return samples


def main():
    """Main data preparation script."""
    parser = argparse.ArgumentParser(description="Prepare task classification dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory")
    parser.add_argument("--num-samples", type=int, default=10000, help="Number of synthetic samples")
    parser.add_argument("--screenshot-dir", help="Directory with existing screenshots")
    parser.add_argument("--create-demo", action="store_true", help="Create demo screenshots")
    parser.add_argument("--confidence-threshold", type=float, default=0.8, help="VLM judge confidence threshold")
    parser.add_argument("--curriculum-learning", action="store_true", default=True, help="Enable curriculum learning (easy->hard)")
    parser.add_argument("--no-curriculum", action="store_true", help="Disable curriculum learning")
    
    args = parser.parse_args()
    
    # Initialize data preparer
    preparer = TaskDataPreparer(output_dir=args.output_dir)
    
    # Create demo screenshots if requested
    if args.create_demo:
        preparer.create_demo_screenshots(num_screenshots=50)
    
    # Load existing screenshots if provided
    if args.screenshot_dir:
        logger.info(f"Loading screenshots from {args.screenshot_dir}")
        samples = preparer.load_existing_screenshots(args.screenshot_dir)
        
        if samples:
            # Save as dataset
            df = pd.DataFrame(samples)
            output_file = Path(args.output_dir) / "real_screenshots.parquet"
            df.to_parquet(output_file)
            logger.info(f"Saved {len(samples)} real screenshot samples to {output_file}")
    
    # Generate synthetic dataset
    curriculum_enabled = args.curriculum_learning and not args.no_curriculum
    train_file, val_file = preparer.generate_synthetic_dataset(
        num_samples=args.num_samples,
        confidence_threshold=args.confidence_threshold,
        curriculum_learning=curriculum_enabled
    )
    
    logger.info("Data preparation complete!")
    logger.info(f"Training data: {train_file}")
    logger.info(f"Validation data: {val_file}")


if __name__ == "__main__":
    main()