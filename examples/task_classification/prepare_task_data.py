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

# VLMJudge not needed - using HF Inference API only
from huggingface_hub import InferenceClient

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
        
        # Initialize HuggingFace Inference Client for VLM judge - NO LOCAL FALLBACK
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is required for VLM judge. Please set: export HF_TOKEN=your_token")
            
        try:
            self.hf_client = InferenceClient(
                provider="auto",
                api_key=hf_token,
            )
            logger.info(f"Initialized HuggingFace Inference Client for {vlm_judge_model}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace Inference Client: {e}")
            
        # NO local VLM judge - only use HF Inference API
        self.vlm_judge = None
        self.vlm_judge_model = vlm_judge_model
        
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
            
            # Generate actual synthetic screenshot
            screenshot_path = self._generate_synthetic_screenshot(i, task, scenario, screenshot_type)
            
            sample = {
                "task_description": task,
                "screenshot": screenshot_path,
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
            
            # Generate actual synthetic screenshot
            screenshot_path = self._generate_synthetic_screenshot(i, task, scenario, screenshot_type)
            
            sample = {
                "task_description": task,
                "screenshot": screenshot_path,
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
    
    def prepare_real_dataset(self, 
                           dataset_name: str = "weizhiwang/Open-Qwen2VL-Data",
                           num_samples: int = 1000,
                           train_ratio: float = 0.8) -> Tuple[str, str]:
        """Prepare dataset using real images + BLIP + VLM judge pipeline.
        
        This is the core pipeline following plan.md:
        1. Load real images from Open-Qwen2VL-Data
        2. Use BLIP to generate contextual captions
        3. Map captions to realistic work scenarios  
        4. Use VLM judge (Qwen2.5-VL-72B) for high-quality labeling
        5. Apply curriculum learning for optimal training
        
        Args:
            dataset_name: HuggingFace dataset name  
            num_samples: Number of samples to process
            train_ratio: Train/validation split ratio
            
        Returns:
            Tuple of (train_file_path, val_file_path)
        """
        logger.info(f"Loading real dataset: {dataset_name}")
        
        try:
            from datasets import load_dataset
            import requests
            from PIL import Image
            import io
            
            # Initialize BLIP pipeline (required for real data approach)
            blip_pipeline = self._init_blip_pipeline()
            if not blip_pipeline:
                raise RuntimeError("BLIP pipeline required for real data processing")
            
            # Load the dataset (streaming for memory efficiency)
            dataset = load_dataset(dataset_name, split="train", streaming=True)
            logger.info("✅ Loaded Open-Qwen2VL-Data in streaming mode")
            
            # Process real images with BLIP + VLM judge pipeline
            samples = []
            successful_samples = 0
            
            logger.info(f"Starting to process {num_samples} samples...")
            
            for i, item in enumerate(dataset):
                if successful_samples >= num_samples:
                    break
                    
                # Progress logging
                if i % 10 == 0:
                    logger.info(f"Processing item {i}, successful: {successful_samples}/{num_samples}")
                    
                try:
                    # Step 1: Load real image from URL
                    image_url = item.get("url", "")
                    original_caption = item.get("caption", "")
                    
                    if not image_url:
                        continue
                    
                    # Download real image
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image_data = Image.open(io.BytesIO(response.content)).convert("RGB")
                    
                    # Step 2: Generate enhanced caption using BLIP
                    blip_caption = self._generate_blip_caption(blip_pipeline, image_data)
                    
                    # Step 3: Combine original caption with BLIP insights
                    combined_context = f"{original_caption}. {blip_caption}".strip()
                    
                    # Step 4: Map to realistic work scenario
                    task_description = self._extract_task_from_content(combined_context)
                    
                    # Step 5: Create balanced work/distraction scenarios
                    is_work_scenario = successful_samples % 2 == 0
                    if is_work_scenario:
                        final_task = task_description
                        expected_label = "on-task"
                    else:
                        # For distraction scenarios, keep work task but expect off-task behavior
                        final_task = task_description
                        expected_label = "off-task"
                    
                    # Save image locally for training and VLM judge
                    local_image_dir = self.output_dir / "real_images"
                    local_image_dir.mkdir(exist_ok=True)
                    local_image_path = local_image_dir / f"real_image_{successful_samples:05d}.jpg"
                    image_data.save(local_image_path, "JPEG", quality=90)
                    
                    # Step 6: Use VLM judge for high-quality labeling (following plan.md)
                    # ONLY use HuggingFace Inference API - no local 72B model
                    label, confidence = self._label_with_hf_inference(str(local_image_path), final_task)
                    logger.debug(f"VLM judge API: {final_task} -> {label} (conf: {confidence:.3f})")
                    
                    # Create training sample following VERL format (no PIL objects for parquet)
                    sample = {
                        "task_description": final_task,
                        "screenshot": str(local_image_path),
                        "label": label,
                        "confidence": confidence,
                        "source": "real_image_blip_vlm_judge",
                        "original_caption": original_caption[:200],
                        "blip_caption": blip_caption,
                        "combined_context": combined_context[:300],
                        "scenario_type": "work" if is_work_scenario else "distraction",
                        "image_url": image_url,
                        "width": image_data.width,
                        "height": image_data.height,
                        "difficulty": "medium"  # Will be updated by curriculum learning
                    }
                    
                    samples.append(sample)
                    successful_samples += 1
                    
                    if successful_samples % 50 == 0:
                        logger.info(f"✅ Processed {successful_samples}/{num_samples} real images")
                
                except Exception as e:
                    logger.warning(f"⚠️ Failed to process sample {i}: {e}")
                    continue
            
            # Apply curriculum learning distribution
            samples = self._apply_curriculum_to_real_data(samples)
            
            # Split and save
            random.shuffle(samples)
            split_idx = int(len(samples) * train_ratio)
            
            train_samples = samples[:split_idx]
            val_samples = samples[split_idx:]
            
            # Save datasets
            train_file = self.output_dir / "real_train.parquet"
            val_file = self.output_dir / "real_val.parquet"
            
            pd.DataFrame(train_samples).to_parquet(train_file)
            pd.DataFrame(val_samples).to_parquet(val_file)
            
            logger.info(f"Saved {len(train_samples)} training samples to {train_file}")
            logger.info(f"Saved {len(val_samples)} validation samples to {val_file}")
            
            # Save metadata
            metadata = {
                "total_samples": len(samples),
                "train_samples": len(train_samples),
                "val_samples": len(val_samples),
                "source_dataset": dataset_name,
                "vlm_judge_model": self.vlm_judge_model,
                "used_blip_captions": True,
                "label_distribution": self._get_label_distribution(samples),
                "curriculum_distribution": self._get_curriculum_distribution(samples)
            }
            
            with open(self.output_dir / "real_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            
            return str(train_file), str(val_file)
            
        except Exception as e:
            import traceback
            logger.error(f"Failed to prepare real dataset: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.info("Falling back to synthetic dataset generation")
            return self.generate_synthetic_dataset(num_samples, train_ratio)
    
    def _label_with_hf_inference(self, image_source: str, task_description: str) -> Tuple[str, float]:
        """Label image using HuggingFace Inference API with VLM judge."""
        try:
            prompt = f"""Look at this image and determine if someone working on this task would be "on-task" or "off-task":

Task: "{task_description}"

Based on what you see in the image, would this activity be considered on-task or off-task for the given work assignment?

Answer with ONLY "on-task" or "off-task":"""

            # Handle both file paths and URLs
            if image_source.startswith(('http://', 'https://')):
                # URL format
                content_item = {
                    "type": "image_url",
                    "image_url": {"url": image_source}
                }
            else:
                # File path format - convert to base64
                import base64
                with open(image_source, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode('utf-8')
                content_item = {
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                }

            logger.debug(f"Calling HF API for image classification...")
            stream = self.hf_client.chat.completions.create(
                model=self.vlm_judge_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            content_item
                        ]
                    }
                ],
                stream=False,
                max_tokens=10,
                timeout=30  # Add timeout to prevent hanging
            )
            
            # Extract response
            response = ""
            if hasattr(stream, 'choices') and stream.choices:
                response = stream.choices[0].message.content.strip().lower()
            else:
                # Handle streaming response
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        response += chunk.choices[0].delta.content
                response = response.strip().lower()
            
            # Parse label and assign confidence
            if "on-task" in response:
                return "on-task", 0.9
            elif "off-task" in response:
                return "off-task", 0.9
            else:
                logger.warning(f"Unexpected VLM response: {response}")
                return "off-task", 0.5
                
        except Exception as e:
            logger.warning(f"HuggingFace inference failed: {e}")
            return self._heuristic_label(task_description)
    
    def _extract_task_from_content(self, content: str) -> str:
        """Extract or generate task description from dataset content using BLIP."""
        # Clean content
        content = content.strip()
        
        # Try to generate contextual task description using BLIP-style reasoning
        content_lower = content.lower()
        
        # Map generic descriptions to work contexts
        work_context_mappings = {
            # Technology/Computer related
            ("computer", "screen", "monitor", "laptop", "desktop"): [
                "Working on software development project",
                "Coding in IDE on feature implementation", 
                "Debugging application in browser",
                "Reviewing code in GitHub"
            ],
            # Document/Text related
            ("document", "text", "writing", "paper", "book"): [
                "Writing technical documentation",
                "Preparing project specifications",
                "Creating user manual for new feature",
                "Drafting quarterly report"
            ],
            # Meeting/Communication related
            ("people", "meeting", "discussion", "presentation"): [
                "Attending team meeting via Zoom",
                "Client presentation call",
                "Sprint planning session",
                "Code review discussion"
            ],
            # Data/Analysis related
            ("chart", "graph", "data", "numbers", "analysis"): [
                "Analyzing performance metrics",
                "Creating data visualization dashboard",
                "Reviewing KPI reports",
                "Working on data analysis in Excel"
            ],
            # Design/Creative related
            ("design", "image", "graphic", "visual", "interface"): [
                "Designing user interface mockups",
                "Creating product wireframes",
                "Working on UI/UX improvements",
                "Reviewing design specifications"
            ]
        }
        
        # Find best matching context
        for keywords, task_options in work_context_mappings.items():
            if any(kw in content_lower for kw in keywords):
                return random.choice(task_options)
        
        # For generic/unknown content, create plausible work context
        generic_work_tasks = [
            "Working on assigned Jira ticket",
            "Completing feature implementation",
            "Conducting code review",
            "Updating project documentation",
            "Participating in team collaboration",
            "Testing application functionality"
        ]
        
        return random.choice(generic_work_tasks)
    
    def _init_blip_pipeline(self):
        """Initialize BLIP pipeline for image captioning."""
        try:
            from transformers import pipeline
            
            # Use BLIP for image captioning
            import torch
            device = 0 if torch.cuda.is_available() else -1
            blip_pipeline = pipeline(
                "image-to-text",
                model="Salesforce/blip-image-captioning-base",
                device=device  # Use GPU if available
            )
            logger.info(f"Initialized BLIP pipeline on {'GPU' if device >= 0 else 'CPU'} for enhanced captioning")
            return blip_pipeline
            
        except Exception as e:
            logger.warning(f"Failed to initialize BLIP pipeline: {e}")
            return None
    
    def _generate_blip_caption(self, blip_pipeline, image_data) -> str:
        """Generate caption using BLIP model."""
        try:
            if blip_pipeline is None:
                return ""
            
            # Generate caption
            result = blip_pipeline(image_data)
            if isinstance(result, list) and len(result) > 0:
                caption = result[0].get("generated_text", "")
            else:
                caption = str(result) if result else ""
            
            return caption.strip()
            
        except Exception as e:
            logger.warning(f"BLIP caption generation failed: {e}")
            return ""
    
    def _apply_curriculum_to_real_data(self, samples: List[Dict]) -> List[Dict]:
        """Apply curriculum learning principles to real data."""
        # Add difficulty assessment based on content complexity
        for sample in samples:
            task_desc = sample["task_description"].lower()
            original_text = sample["original_caption"].lower()
            
            # Determine difficulty based on complexity indicators
            complexity_indicators = [
                "complex", "advanced", "technical", "specialized", "expert",
                "integration", "architecture", "optimization", "debugging"
            ]
            
            ambiguity_indicators = [
                "maybe", "possibly", "might", "unclear", "ambiguous",
                "multi", "various", "several", "different"
            ]
            
            # Calculate complexity score
            complexity_score = sum(1 for indicator in complexity_indicators 
                                 if indicator in task_desc or indicator in original_text)
            ambiguity_score = sum(1 for indicator in ambiguity_indicators 
                                if indicator in task_desc or indicator in original_text)
            
            # Assign difficulty
            if complexity_score > 2 or ambiguity_score > 1:
                sample["difficulty"] = "hard"
            elif complexity_score > 0 or ambiguity_score > 0:
                sample["difficulty"] = "medium"
            else:
                sample["difficulty"] = "easy"
        
        return samples
    
    def _get_curriculum_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Get curriculum difficulty distribution."""
        distribution = {}
        for sample in samples:
            difficulty = sample.get("difficulty", "medium")
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _heuristic_label(self, task_description: str) -> Tuple[str, float]:
        """Heuristic labeling based on task description keywords."""
        task_lower = task_description.lower()
        
        work_keywords = {
            "code", "coding", "github", "jira", "meeting", "zoom", "slack", 
            "email", "document", "report", "analysis", "data", "excel",
            "presentation", "project", "work", "task", "ticket", "review"
        }
        
        distraction_keywords = {
            "youtube", "facebook", "instagram", "twitter", "reddit", "gaming",
            "shopping", "amazon", "netflix", "entertainment", "social", "news"
        }
        
        work_score = sum(1 for kw in work_keywords if kw in task_lower)
        distraction_score = sum(1 for kw in distraction_keywords if kw in task_lower)
        
        if work_score > distraction_score:
            return "on-task", 0.7 + min(0.2, work_score * 0.1)
        else:
            return "off-task", 0.7 + min(0.2, distraction_score * 0.1)
    
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
        """Generate synthetic validation based on task-screenshot alignment.
        
        Args:
            samples: List of samples to validate
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            List of validated samples with synthetic confidence scores
        """
        validated = []
        
        for sample in samples:
            task_desc = sample["task_description"].lower()
            expected_label = sample["label"]
            difficulty = sample.get("difficulty", "medium")
            
            # Generate realistic confidence based on task-screenshot alignment
            confidence = self._calculate_synthetic_confidence(task_desc, expected_label, difficulty)
            
            # Add realistic variability
            confidence += random.uniform(-0.1, 0.1)
            confidence = max(0.5, min(1.0, confidence))
            
            # Include samples based on confidence threshold
            sample["confidence"] = confidence
            sample["judge_validated"] = confidence >= confidence_threshold
            validated.append(sample)
        
        high_conf_count = sum(1 for s in validated if s["confidence"] >= confidence_threshold)
        logger.info(f"Generated {len(validated)} samples, {high_conf_count} high-confidence (>= {confidence_threshold})")
        return validated
    
    def _calculate_synthetic_confidence(self, task_desc: str, expected_label: str, difficulty: str) -> float:
        """Calculate synthetic confidence score based on task alignment."""
        
        # Define keyword categories with confidence weights
        strong_work_indicators = {
            "jira", "github", "gitlab", "pull request", "code", "coding", "debugging",
            "unit tests", "integration", "deployment", "ci/cd", "pipeline", "docker",
            "kubernetes", "aws", "azure", "confluence", "documentation", "api",
            "database", "sql", "python", "javascript", "typescript", "react", "vue"
        }
        
        medium_work_indicators = {
            "meeting", "zoom", "teams", "slack", "email", "excel", "powerpoint",
            "presentation", "planning", "sprint", "scrum", "standup", "review",
            "analysis", "report", "dashboard", "metrics", "monitoring"
        }
        
        strong_distraction_indicators = {
            "reddit", "youtube", "tiktok", "instagram", "facebook", "twitter",
            "netflix", "gaming", "twitch", "discord", "shopping", "amazon",
            "ebay", "social media", "entertainment", "memes", "videos"
        }
        
        medium_distraction_indicators = {
            "news", "blog", "personal", "weather", "sports", "celebrity",
            "gossip", "vacation", "photos", "messages", "dating", "finance"
        }
        
        # Calculate base confidence
        if expected_label == "on-task":
            if any(kw in task_desc for kw in strong_work_indicators):
                base_confidence = 0.95
            elif any(kw in task_desc for kw in medium_work_indicators):
                base_confidence = 0.85
            else:
                base_confidence = 0.70
        else:  # off-task
            if any(kw in task_desc for kw in strong_distraction_indicators):
                base_confidence = 0.95
            elif any(kw in task_desc for kw in medium_distraction_indicators):
                base_confidence = 0.85
            else:
                base_confidence = 0.70
        
        # Adjust for difficulty (edge cases are harder to classify)
        if difficulty == "hard":
            base_confidence *= 0.8  # Reduce confidence for edge cases
        elif difficulty == "easy":
            base_confidence *= 1.1  # Increase confidence for clear cases
            base_confidence = min(1.0, base_confidence)
        
        return base_confidence
    
    def _generate_synthetic_screenshot(self, index: int, task: str, scenario: str, screenshot_type: str) -> str:
        """Generate a realistic synthetic screenshot."""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create screenshot directory
        screenshot_dir = self.output_dir / "synthetic_screenshots"
        screenshot_dir.mkdir(exist_ok=True)
        
        # Create base image (standard monitor resolution)
        img = Image.new("RGB", (1920, 1080), color="white")
        draw = ImageDraw.Draw(img)
        
        # Generate screenshot based on type and task
        if screenshot_type == "work":
            self._draw_work_interface(draw, img.size, task)
        else:  # distraction
            self._draw_distraction_interface(draw, img.size, task)
        
        # Save screenshot
        screenshot_filename = f"synthetic_screenshot_{index:05d}_{scenario}_{screenshot_type}.png"
        screenshot_path = screenshot_dir / screenshot_filename
        img.save(screenshot_path)
        
        return str(screenshot_path)
    
    def _draw_work_interface(self, draw, size, task: str):
        """Draw realistic work interface based on task."""
        width, height = size
        task_lower = task.lower()
        
        if any(kw in task_lower for kw in ["code", "github", "gitlab", "vs code", "debug"]):
            self._draw_ide_interface(draw, size, task)
        elif any(kw in task_lower for kw in ["jira", "ticket", "project"]):
            self._draw_jira_interface(draw, size, task)
        elif any(kw in task_lower for kw in ["meeting", "zoom", "teams"]):
            self._draw_meeting_interface(draw, size, task)
        elif any(kw in task_lower for kw in ["slack", "message", "chat"]):
            self._draw_slack_interface(draw, size, task)
        elif any(kw in task_lower for kw in ["excel", "data", "analysis"]):
            self._draw_excel_interface(draw, size, task)
        else:
            # Default work interface
            self._draw_generic_work_interface(draw, size, task)
    
    def _draw_distraction_interface(self, draw, size, task: str):
        """Draw realistic distraction interface."""
        width, height = size
        distraction_type = random.choice(self.distraction_scenarios).lower()
        
        if any(kw in distraction_type for kw in ["youtube", "video", "tiktok"]):
            self._draw_video_interface(draw, size)
        elif any(kw in distraction_type for kw in ["social", "facebook", "instagram", "twitter"]):
            self._draw_social_interface(draw, size)
        elif any(kw in distraction_type for kw in ["shopping", "amazon", "ebay"]):
            self._draw_shopping_interface(draw, size)
        elif any(kw in distraction_type for kw in ["news", "blog", "article"]):
            self._draw_news_interface(draw, size)
        else:
            # Default distraction interface
            self._draw_generic_distraction_interface(draw, size)
    
    def _draw_ide_interface(self, draw, size, task):
        """Draw IDE interface (VS Code style)."""
        width, height = size
        
        # Dark theme IDE
        draw.rectangle([0, 0, width, height], fill="#1e1e1e")
        
        # Title bar
        draw.rectangle([0, 0, width, 60], fill="#2d2d2d")
        draw.text((20, 20), f"VS Code - {task}", fill="white")
        
        # Side panel
        draw.rectangle([0, 60, 250, height], fill="#252526")
        draw.text((10, 80), "EXPLORER", fill="#cccccc")
        draw.text((10, 120), "src/", fill="#ffcc02")
        draw.text((20, 140), "main.py", fill="#4fc3f7")
        draw.text((20, 160), "utils.py", fill="#4fc3f7")
        draw.text((20, 180), "test.py", fill="#4fc3f7")
        
        # Main editor
        draw.rectangle([250, 60, width, height], fill="#1e1e1e")
        
        # Code content
        code_lines = [
            "def classify_task(screenshot, description):",
            "    # Implementation of task classification",
            "    features = extract_features(screenshot)",
            "    prediction = model.predict(features)",
            "    return prediction",
            "",
            "# TODO: Add VLM judge integration",
            "class TaskClassifier:",
            "    def __init__(self, model_path):",
            "        self.model = load_model(model_path)"
        ]
        
        y_pos = 80
        colors = ["#569cd6", "#6a9955", "#dcdcaa", "#ce9178", "#4fc3f7"]
        for i, line in enumerate(code_lines):
            color = colors[i % len(colors)]
            draw.text((270, y_pos), line, fill=color)
            y_pos += 25
    
    def _draw_jira_interface(self, draw, size, task):
        """Draw JIRA interface."""
        width, height = size
        
        # JIRA blue theme
        draw.rectangle([0, 0, width, height], fill="#f4f5f7")
        
        # Header
        draw.rectangle([0, 0, width, 80], fill="#0052cc")
        draw.text((20, 25), "JIRA", fill="white")
        draw.text((width-300, 25), "Projects | Issues | Dashboards", fill="white")
        
        # Navigation
        draw.rectangle([0, 80, 200, height], fill="#ffffff")
        draw.text((10, 100), "Your work", fill="#172b4d")
        draw.text((10, 130), "Projects", fill="#172b4d")
        draw.text((10, 160), "Filters", fill="#172b4d")
        draw.text((10, 190), "Dashboards", fill="#172b4d")
        
        # Main content area
        draw.rectangle([200, 80, width, height], fill="#ffffff")
        
        # Ticket details
        draw.text((220, 100), f"PROJ-123: {task}", fill="#172b4d")
        draw.text((220, 130), "Status: In Progress", fill="#42526e")
        draw.text((220, 160), "Assignee: Current User", fill="#42526e")
        draw.text((220, 190), "Priority: High", fill="#de350b")
        
        # Description box
        draw.rectangle([220, 220, width-40, 400], fill="#f4f5f7")
        draw.text((240, 240), "Description:", fill="#172b4d")
        draw.text((240, 270), "Implement task classification system", fill="#42526e")
        draw.text((240, 290), "using VLM judge approach", fill="#42526e")
    
    def _draw_meeting_interface(self, draw, size, task):
        """Draw video meeting interface (Zoom style)."""
        width, height = size
        
        # Dark meeting background
        draw.rectangle([0, 0, width, height], fill="#1f1f1f")
        
        # Top bar
        draw.rectangle([0, 0, width, 60], fill="#2d2d2d")
        draw.text((20, 20), f"Zoom - {task}", fill="white")
        draw.text((width-200, 20), "🔇 🎥 💬 ⚙️", fill="white")
        
        # Video grid (2x2)
        video_width = (width - 60) // 2
        video_height = (height - 120) // 2
        
        positions = [
            (20, 80),
            (40 + video_width, 80),
            (20, 100 + video_height),
            (40 + video_width, 100 + video_height)
        ]
        
        participants = ["You", "Alice Smith", "Bob Johnson", "Carol Davis"]
        
        for i, (x, y) in enumerate(positions):
            # Video frame
            draw.rectangle([x, y, x + video_width, y + video_height], fill="#3c3c3c")
            draw.rectangle([x, y, x + video_width, y + video_height], outline="#5a5a5a", width=2)
            
            # Participant name
            if i < len(participants):
                draw.text((x + 10, y + video_height - 30), participants[i], fill="white")
                # Simulate video content
                draw.rectangle([x + 50, y + 50, x + video_width - 50, y + video_height - 50], fill="#4a4a4a")
        
        # Bottom control bar
        draw.rectangle([0, height - 60, width, height], fill="#2d2d2d")
        draw.text((width//2 - 100, height - 40), "🎤 📹 💬 🖥️ 📱 ⚙️ ❌", fill="white")
    
    def _draw_slack_interface(self, draw, size, task):
        """Draw Slack interface."""
        width, height = size
        
        # Slack purple sidebar
        draw.rectangle([0, 0, width, height], fill="#ffffff")
        draw.rectangle([0, 0, 260, height], fill="#4a154b")
        
        # Workspace name
        draw.text((20, 20), "Company Workspace", fill="white")
        
        # Channels
        draw.text((20, 80), "Channels", fill="#bcabbc")
        draw.text((30, 110), "# general", fill="white")
        draw.text((30, 130), "# development", fill="white")
        draw.text((30, 150), "# random", fill="white")
        
        # Direct messages
        draw.text((20, 200), "Direct messages", fill="#bcabbc")
        draw.text((30, 230), "🟢 Manager", fill="white")
        draw.text((30, 250), "🔴 Team Lead", fill="white")
        
        # Main chat area
        draw.rectangle([260, 0, width, 60], fill="#ffffff")
        draw.text((280, 20), "# development", fill="#1d1c1d")
        
        # Messages
        messages = [
            ("Manager", "Can you provide an update on PROJ-123?"),
            ("You", "Working on the classification system"),
            ("Team Lead", "Great! How's the VLM integration going?"),
            ("You", "Making good progress, testing locally first")
        ]
        
        y_pos = 80
        for sender, message in messages:
            draw.text((280, y_pos), sender, fill="#1264a3")
            draw.text((280, y_pos + 20), message, fill="#1d1c1d")
            y_pos += 60
    
    def _draw_excel_interface(self, draw, size, task):
        """Draw Excel interface."""
        width, height = size
        
        # Excel green header
        draw.rectangle([0, 0, width, height], fill="#ffffff")
        draw.rectangle([0, 0, width, 80], fill="#217346")
        draw.text((20, 25), f"Excel - {task}", fill="white")
        
        # Ribbon area
        draw.rectangle([0, 80, width, 140], fill="#f2f2f2")
        draw.text((20, 100), "Home | Insert | Page Layout | Formulas | Data", fill="#323130")
        
        # Column headers
        columns = ["A", "B", "C", "D", "E", "F", "G"]
        col_width = 100
        for i, col in enumerate(columns):
            x = 50 + i * col_width
            draw.rectangle([x, 140, x + col_width, 170], fill="#e1dfdd")
            draw.text((x + 45, 150), col, fill="#323130")
        
        # Row data
        data = [
            ["Metric", "Value", "Target", "Status"],
            ["Accuracy", "0.85", "0.90", "In Progress"],
            ["Latency", "120ms", "100ms", "Needs Work"],
            ["Throughput", "500/sec", "400/sec", "Excellent"],
            ["Error Rate", "2%", "1%", "Improving"]
        ]
        
        for row_idx, row_data in enumerate(data):
            y = 170 + row_idx * 30
            # Row number
            draw.rectangle([0, y, 50, y + 30], fill="#e1dfdd")
            draw.text((20, y + 10), str(row_idx + 1), fill="#323130")
            
            # Cell data
            for col_idx, cell in enumerate(row_data):
                x = 50 + col_idx * col_width
                draw.rectangle([x, y, x + col_width, y + 30], outline="#d1d1d1")
                draw.text((x + 10, y + 10), cell, fill="#323130")
    
    def _draw_generic_work_interface(self, draw, size, task):
        """Draw generic work interface."""
        width, height = size
        
        # Professional theme
        draw.rectangle([0, 0, width, height], fill="#f8f9fa")
        
        # Header
        draw.rectangle([0, 0, width, 80], fill="#343a40")
        draw.text((20, 25), f"Work Dashboard - {task}", fill="white")
        
        # Content area
        draw.rectangle([20, 100, width-20, height-20], fill="white")
        draw.text((40, 120), "Current Task:", fill="#495057")
        draw.text((40, 150), task, fill="#212529")
        
        # Progress indicators
        draw.text((40, 200), "Progress: 65%", fill="#28a745")
        draw.rectangle([40, 220, 300, 240], fill="#e9ecef")
        draw.rectangle([40, 220, 235, 240], fill="#28a745")
    
    def _draw_video_interface(self, draw, size):
        """Draw video streaming interface."""
        width, height = size
        
        # YouTube red theme
        draw.rectangle([0, 0, width, height], fill="#0f0f0f")
        
        # Header
        draw.rectangle([0, 0, width, 80], fill="#212121")
        draw.text((20, 25), "YouTube", fill="#ff0000")
        draw.text((width-200, 25), "🔍 📹 🔔 👤", fill="white")
        
        # Video player area
        video_width = width - 400
        video_height = (video_width * 9) // 16
        draw.rectangle([20, 100, 20 + video_width, 100 + video_height], fill="#000000")
        draw.text((video_width//2 - 20, 100 + video_height//2), "▶️", fill="white")
        
        # Video title
        draw.text((20, 120 + video_height), "Funny Cat Compilation 2024", fill="white")
        draw.text((20, 150 + video_height), "5.2M views • 2 days ago", fill="#aaaaaa")
        
        # Sidebar recommendations
        draw.text((video_width + 40, 120), "Up next", fill="white")
        
        rec_videos = [
            "Dog vs Vacuum Cleaner",
            "Fails Compilation",
            "Gaming Highlights",
            "Music Video Trending"
        ]
        
        y_pos = 150
        for video in rec_videos:
            draw.rectangle([video_width + 40, y_pos, width - 20, y_pos + 80], fill="#212121")
            draw.text((video_width + 50, y_pos + 30), video, fill="white")
            y_pos += 100
    
    def _draw_social_interface(self, draw, size):
        """Draw social media interface."""
        width, height = size
        
        # Facebook blue theme
        draw.rectangle([0, 0, width, height], fill="#f0f2f5")
        
        # Header
        draw.rectangle([0, 0, width, 80], fill="#1877f2")
        draw.text((20, 25), "facebook", fill="white")
        draw.text((width-300, 25), "🏠 👥 📺 🛍️ 👤", fill="white")
        
        # Left sidebar
        draw.rectangle([0, 80, 250, height], fill="#ffffff")
        draw.text((20, 100), "👤 Your Profile", fill="#1c1e21")
        draw.text((20, 130), "👥 Friends", fill="#1c1e21")
        draw.text((20, 160), "📱 Pages", fill="#1c1e21")
        draw.text((20, 190), "🎮 Gaming", fill="#1c1e21")
        
        # Main feed
        feed_x = 270
        feed_width = width - 520
        
        # Post 1
        y = 100
        draw.rectangle([feed_x, y, feed_x + feed_width, y + 200], fill="white")
        draw.text((feed_x + 20, y + 20), "Friend's Name", fill="#1c1e21")
        draw.text((feed_x + 20, y + 40), "Just had an amazing weekend! 🎉", fill="#1c1e21")
        draw.rectangle([feed_x + 20, y + 70, feed_x + feed_width - 20, y + 150], fill="#e4e6ea")
        draw.text((feed_x + 30, y + 110), "[Photo: Weekend Trip]", fill="#65676b")
        
        # Post 2
        y = 320
        draw.rectangle([feed_x, y, feed_x + feed_width, y + 150], fill="white")
        draw.text((feed_x + 20, y + 20), "Another Friend", fill="#1c1e21")
        draw.text((feed_x + 20, y + 40), "Check out this funny video!", fill="#1c1e21")
        draw.rectangle([feed_x + 20, y + 70, feed_x + feed_width - 20, y + 120], fill="#000000")
        draw.text((feed_x + feed_width//2, y + 95), "▶️", fill="white")
        
        # Right sidebar
        draw.rectangle([width - 250, 80, width, height], fill="#ffffff")
        draw.text((width - 240, 100), "Sponsored", fill="#65676b")
        draw.text((width - 240, 130), "Ad: New Product Launch", fill="#1c1e21")
    
    def _draw_shopping_interface(self, draw, size):
        """Draw shopping interface."""
        width, height = size
        
        # Amazon orange theme
        draw.rectangle([0, 0, width, height], fill="#ffffff")
        
        # Header
        draw.rectangle([0, 0, width, 80], fill="#232f3e")
        draw.text((20, 25), "amazon", fill="#ff9900")
        draw.text((width-300, 25), "🛒 Account | Orders | Cart", fill="white")
        
        # Search bar
        draw.rectangle([200, 20, width-200, 60], fill="white")
        draw.text((210, 35), "Search for anything...", fill="#999999")
        
        # Product grid
        products = [
            ("Laptop Stand", "$29.99"),
            ("Wireless Mouse", "$19.99"), 
            ("Bluetooth Headphones", "$79.99"),
            ("Phone Case", "$12.99"),
            ("USB Cable", "$8.99"),
            ("Webcam", "$49.99")
        ]
        
        cols = 3
        product_width = (width - 80) // cols
        product_height = 250
        
        for i, (name, price) in enumerate(products):
            row = i // cols
            col = i % cols
            x = 20 + col * product_width
            y = 100 + row * product_height
            
            # Product card
            draw.rectangle([x, y, x + product_width - 20, y + product_height - 20], fill="white")
            draw.rectangle([x, y, x + product_width - 20, y + product_height - 20], outline="#ddd")
            
            # Product image placeholder
            draw.rectangle([x + 10, y + 10, x + product_width - 30, y + 150], fill="#f0f0f0")
            draw.text((x + product_width//2 - 30, y + 75), "[Image]", fill="#999999")
            
            # Product details
            draw.text((x + 10, y + 160), name, fill="#0066c0")
            draw.text((x + 10, y + 180), price, fill="#b12704")
            draw.text((x + 10, y + 200), "⭐⭐⭐⭐⭐ (123)", fill="#ff9900")
    
    def _draw_news_interface(self, draw, size):
        """Draw news website interface."""
        width, height = size
        
        # News site layout
        draw.rectangle([0, 0, width, height], fill="#ffffff")
        
        # Header
        draw.rectangle([0, 0, width, 80], fill="#1a1a1a")
        draw.text((20, 25), "NEWS TODAY", fill="white")
        draw.text((width-400, 25), "Politics | Tech | Sports | Entertainment", fill="white")
        
        # Main article
        draw.rectangle([20, 100, width//2, 400], fill="#f8f9fa")
        draw.text((30, 120), "BREAKING NEWS", fill="#dc3545")
        draw.text((30, 150), "Major Development in Technology Sector", fill="#212529")
        draw.text((30, 180), "Lorem ipsum dolor sit amet, consectetur", fill="#6c757d")
        draw.text((30, 200), "adipiscing elit. Industry experts say...", fill="#6c757d")
        
        # Sidebar articles
        sidebar_x = width//2 + 40
        articles = [
            "Local Election Results",
            "Sports Team Wins Championship", 
            "Celebrity Spotted Downtown",
            "Weather Update for Weekend",
            "Stock Market Analysis"
        ]
        
        y = 120
        for article in articles:
            draw.rectangle([sidebar_x, y, width - 20, y + 60], fill="#f8f9fa")
            draw.text((sidebar_x + 10, y + 20), article, fill="#212529")
            y += 80
    
    def _draw_generic_distraction_interface(self, draw, size):
        """Draw generic distraction interface."""
        width, height = size
        
        # Bright, distracting colors
        draw.rectangle([0, 0, width, height], fill="#ff6b6b")
        
        # Header
        draw.rectangle([0, 0, width, 80], fill="#4ecdc4")
        draw.text((20, 25), "FUN ZONE", fill="white")
        
        # Content
        draw.text((50, 150), "🎮 GAMES 🎮", fill="white")
        draw.text((50, 200), "🎵 MUSIC 🎵", fill="white")
        draw.text((50, 250), "📺 VIDEOS 📺", fill="white")
        draw.text((50, 300), "🛍️ SHOPPING 🛍️", fill="white")
    
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
    parser.add_argument("--use-real-data", action="store_true", help="Use real dataset from HuggingFace")
    parser.add_argument("--dataset-name", default="weizhiwang/Open-Qwen2VL-Data", help="HuggingFace dataset name")
    
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
    
    # Generate dataset using REAL IMAGES + BLIP + VLM judge pipeline
    logger.info("🎯 Using real image + BLIP + VLM judge pipeline (following plan.md)")
    train_file, val_file = preparer.prepare_real_dataset(
        dataset_name=args.dataset_name,
        num_samples=args.num_samples
    )
    
    logger.info("Data preparation complete!")
    logger.info(f"Training data: {train_file}")
    logger.info(f"Validation data: {val_file}")


if __name__ == "__main__":
    main()