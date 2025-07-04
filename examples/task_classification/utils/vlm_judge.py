"""VLM Judge for Task Classification

Implements Seed 1.5's VLM-as-reward-model approach for binary task classification.
Uses Qwen2.5-VL-72B to provide high-quality verification of student model predictions.
"""

import json
import re
import time
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image
except ImportError as e:
    logging.warning(f"Missing dependencies for VLM judge: {e}")

logger = logging.getLogger(__name__)


class VLMJudge:
    """VLM Judge using Qwen2.5-VL-72B for task classification verification.
    
    Based on Seed 1.5's approach of using VLM as generative classifier.
    Computes preference probabilities directly from logits.
    """
    
    def __init__(self, 
                 model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct",
                 device: str = "auto",
                 cache_size: int = 10000,
                 tensor_parallel_size: int = 4):
        """Initialize VLM judge.
        
        Args:
            model_path: Path to Qwen2.5-VL model
            device: Device for inference
            cache_size: Maximum cache entries
            tensor_parallel_size: Number of GPUs for model parallel
        """
        self.model_path = model_path
        self.device = device
        self.cache_size = cache_size
        self.tensor_parallel_size = tensor_parallel_size
        
        # Judgment cache for efficiency
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Model components (lazy loaded)
        self._model = None
        self._processor = None
        self._tokenizer = None
        
        # Seed 1.5 approach: preference probability computation
        self.on_task_tokens = ["on-task", "on_task", "ontask", "on", "yes", "true"]
        self.off_task_tokens = ["off-task", "off_task", "offtask", "off", "no", "false"]
        
    def _load_model(self):
        """Lazy load the VLM model."""
        if self._model is not None:
            return
            
        logger.info(f"Loading VLM judge: {self.model_path}")
        try:
            # Load model with appropriate settings for 72B
            self._model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
                attn_implementation="flash_attention_2"
            )
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            logger.info("VLM judge loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load VLM judge: {e}")
            raise
    
    def _create_cache_key(self, screenshot_path: str, task_description: str) -> str:
        """Create cache key for judgment."""
        return f"{screenshot_path}||{task_description}"
    
    def judge_classification(self, 
                           screenshot_path: str, 
                           task_description: str,
                           use_cache: bool = True) -> Tuple[str, float]:
        """Judge binary classification using VLM.
        
        Args:
            screenshot_path: Path to screenshot image
            task_description: Description of the assigned task
            use_cache: Whether to use cached results
            
        Returns:
            Tuple of (classification, confidence)
        """
        # Check cache first
        cache_key = self._create_cache_key(screenshot_path, task_description)
        if use_cache and cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        # Load model if needed
        self._load_model()
        
        try:
            # Load and process image
            image = Image.open(screenshot_path).convert("RGB")
            
            # Create judge prompt based on Seed 1.5 approach
            prompt = self._create_judge_prompt(task_description)
            
            # Get VLM classification using preference probabilities
            classification, confidence = self._get_vlm_classification(image, prompt)
            
            # Cache result
            if use_cache and len(self._cache) < self.cache_size:
                self._cache[cache_key] = (classification, confidence)
            
            return classification, confidence
            
        except Exception as e:
            logger.error(f"VLM judge failed for {screenshot_path}: {e}")
            return "off-task", 0.5  # Conservative default
    
    def _create_judge_prompt(self, task_description: str) -> str:
        """Create prompt for VLM judge based on Seed 1.5 approach."""
        prompt = f"""You are an expert at analyzing screenshots to determine if a user is on-task or off-task.

Task assigned to user: {task_description}

Analyze this screenshot and determine if the user is currently on-task or off-task.

Consider these factors:
1. Is the visible application/website relevant to the assigned task?
2. Is the content being viewed/edited related to the task?
3. Are there any distracting elements or activities visible?
4. Does the overall screen context suggest focus on the assigned task?

Guidelines:
- ON-TASK: User is actively working on or engaged with the assigned task
- OFF-TASK: User is doing something unrelated to the assigned task

Respond with exactly one word: on-task or off-task"""
        
        return prompt
    
    def _get_vlm_classification(self, image: Image.Image, prompt: str) -> Tuple[str, float]:
        """Get classification using Seed 1.5's preference probability approach."""
        try:
            # Prepare inputs
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # Apply chat template
            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = self._processor(
                text=[text], 
                images=[image], 
                padding=True, 
                return_tensors="pt"
            ).to(self._model.device)
            
            # Generate with logits output
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )
            
            # Decode response
            generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
            response = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip().lower()
            
            # Compute preference probabilities from logits (Seed 1.5 approach)
            if hasattr(outputs, 'scores') and len(outputs.scores) > 0:
                first_token_logits = outputs.scores[0][0]  # First generated token logits
                probs = torch.softmax(first_token_logits, dim=-1)
                
                # Get probabilities for on-task vs off-task tokens
                on_task_prob = 0.0
                off_task_prob = 0.0
                
                for token in self.on_task_tokens:
                    token_id = self._tokenizer.encode(token, add_special_tokens=False)
                    if token_id:
                        on_task_prob += probs[token_id[0]].item()
                
                for token in self.off_task_tokens:
                    token_id = self._tokenizer.encode(token, add_special_tokens=False)
                    if token_id:
                        off_task_prob += probs[token_id[0]].item()
                
                # Normalize probabilities
                total_prob = on_task_prob + off_task_prob
                if total_prob > 0:
                    on_task_prob /= total_prob
                    off_task_prob /= total_prob
                    
                    # Choose classification and confidence
                    if on_task_prob > off_task_prob:
                        classification = "on-task"
                        confidence = on_task_prob
                    else:
                        classification = "off-task"
                        confidence = off_task_prob
                else:
                    # Fallback to response parsing
                    classification, confidence = self._parse_response(response)
            else:
                # Fallback to response parsing
                classification, confidence = self._parse_response(response)
            
            return classification, confidence
            
        except Exception as e:
            logger.error(f"VLM classification failed: {e}")
            return "off-task", 0.5
    
    def _parse_response(self, response: str) -> Tuple[str, float]:
        """Parse VLM response to extract classification."""
        response = response.lower().strip()
        
        # Check for clear indicators
        if "on-task" in response or "on_task" in response:
            return "on-task", 0.8
        elif "off-task" in response or "off_task" in response:
            return "off-task", 0.8
        elif "on" in response[:5]:  # "on" at beginning
            return "on-task", 0.6
        elif "off" in response[:5]:  # "off" at beginning
            return "off-task", 0.6
        else:
            # Conservative default
            return "off-task", 0.5
    
    def batch_judge(self, 
                   screenshot_task_pairs: List[Tuple[str, str]],
                   use_cache: bool = True) -> List[Tuple[str, float]]:
        """Batch judge multiple screenshot-task pairs.
        
        Args:
            screenshot_task_pairs: List of (screenshot_path, task_description) tuples
            use_cache: Whether to use cached results
            
        Returns:
            List of (classification, confidence) tuples
        """
        results = []
        
        for screenshot_path, task_description in screenshot_task_pairs:
            classification, confidence = self.judge_classification(
                screenshot_path, task_description, use_cache
            )
            results.append((classification, confidence))
        
        return results
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_cache_size": self.cache_size
        }
    
    def clear_cache(self):
        """Clear judgment cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0