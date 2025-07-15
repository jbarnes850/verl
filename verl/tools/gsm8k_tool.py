# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import json
import asyncio
from typing import Any, Optional, Tuple
from uuid import uuid4
from pathlib import Path
import sys

from verl.utils.reward_score import gsm8k

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# Add CRMArena to path
crm_arena_path = Path(__file__).parent.parent.parent / "CRMArena"
if str(crm_arena_path) not in sys.path:
    sys.path.append(str(crm_arena_path))

# Import CRMArena functions
try:
    import sys
    import os
    crm_path = os.path.join(os.path.dirname(__file__), '../../CRMArena')
    if os.path.exists(crm_path):
        sys.path.insert(0, crm_path)
    from test_functions.functions import *
    CRM_FUNCTIONS_AVAILABLE = True
    crm_functions = sys.modules['test_functions.functions']
except ImportError:
    CRM_FUNCTIONS_AVAILABLE = False
    crm_functions = None


class Gsm8kTool(BaseTool):
    """A demo tool for calculating the reward of gsm8k.

    - `to_openai_function_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.database_type = config.get("database_type", "gsm8k")
        
        # Map of CRMArena function names to actual functions
        if CRM_FUNCTIONS_AVAILABLE and self.database_type == "crmarena":
            self.crm_function_map = {
                "get_agents_with_max_cases": crm_functions.get_agents_with_max_cases,
                "get_agents_with_min_cases": crm_functions.get_agents_with_min_cases,
                "calculate_average_handle_time": crm_functions.calculate_average_handle_time,
                "get_start_date": crm_functions.get_start_date,
                "get_period": crm_functions.get_period,
                "get_agent_handled_cases_by_period": crm_functions.get_agent_handled_cases_by_period,
                "get_qualified_agent_ids_by_case_count": crm_functions.get_qualified_agent_ids_by_case_count,
                "get_cases": crm_functions.get_cases,
                "get_non_transferred_case_ids": crm_functions.get_non_transferred_case_ids,
                "get_agent_transferred_cases_by_period": crm_functions.get_agent_transferred_cases_by_period,
                "get_shipping_state": crm_functions.get_shipping_state,
                "calculate_region_average_closure_times": crm_functions.calculate_region_average_closure_times,
                "get_order_item_ids_by_product": crm_functions.get_order_item_ids_by_product,
                "get_issue_counts": crm_functions.get_issue_counts,
                "find_id_with_max_value": crm_functions.find_id_with_max_value,
                "find_id_with_min_value": crm_functions.find_id_with_min_value,
                "get_account_id_by_contact_id": crm_functions.get_account_id_by_contact_id,
                "get_purchase_history": crm_functions.get_purchase_history,
                "get_month_to_case_count": crm_functions.get_month_to_case_count,
                "search_knowledge_articles": crm_functions.search_knowledge_articles,
                "search_products": crm_functions.search_products,
                "get_issues": crm_functions.get_issues,
                "get_email_messages_by_case_id": crm_functions.get_email_messages_by_case_id,
                "get_livechat_transcript_by_case_id": crm_functions.get_livechat_transcript_by_case_id,
                "submit": crm_functions.submit
            }
            
            # Initialize Salesforce connection
            try:
                from dotenv import load_dotenv
                import os
                load_dotenv('CRMArena/.env')
                
                from simple_salesforce import Salesforce
                org_type = config.get("org_type", "original")
                
                if org_type == "original":
                    username = os.getenv("SALESFORCE_USERNAME")
                    password = os.getenv("SALESFORCE_PASSWORD")
                    security_token = os.getenv("SALESFORCE_SECURITY_TOKEN")
                elif org_type == "b2b":
                    username = os.getenv("SALESFORCE_B2B_USERNAME")
                    password = os.getenv("SALESFORCE_B2B_PASSWORD")
                    security_token = os.getenv("SALESFORCE_B2B_SECURITY_TOKEN")
                else:
                    username = os.getenv("SALESFORCE_B2C_USERNAME")
                    password = os.getenv("SALESFORCE_B2C_PASSWORD")
                    security_token = os.getenv("SALESFORCE_B2C_SECURITY_TOKEN")
                
                # Set global sf for CRMArena functions
                crm_functions.sf = Salesforce(username=username, password=password, security_token=security_token)
                self.sf_initialized = True
            except Exception as e:
                print(f"Warning: Could not initialize Salesforce connection: {e}")
                self.sf_initialized = False
        else:
            self.crm_function_map = {}
            self.sf_initialized = False

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, ground_truth: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": 0.0,
        }
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        if self.database_type == "crmarena" and self.crm_function_map:
            # Handle CRMArena database queries
            func_name = parameters.get("function_name")
            func_args = parameters.get("arguments", {})
            
            if not func_name:
                error_msg = "Error: function_name not provided in parameters"
                return error_msg, 0.0, {"error": True}
            
            if func_name not in self.crm_function_map:
                error_msg = f"Error: Unknown function '{func_name}'"
                return error_msg, 0.0, {"error": True, "unknown_function": func_name}
            
            try:
                # Log the tool call
                print(f"\n[TOOL CALL] Function: {func_name}")
                print(f"[TOOL CALL] Arguments: {func_args}")
                
                # Execute the CRMArena function
                func = self.crm_function_map[func_name]
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: func(**func_args))
                
                # Convert result to string format
                if isinstance(result, str) and result.startswith("Error:"):
                    print(f"[TOOL CALL] Error result: {result[:100]}...")
                    return result, 0.0, {"error": True, "function": func_name}
                
                result_str = json.dumps(result) if not isinstance(result, str) else result
                print(f"[TOOL CALL] Success! Result length: {len(result_str)} chars")
                
                # Log a preview of the result for debugging
                if len(result_str) > 200:
                    print(f"[TOOL CALL] Result preview: {result_str[:200]}...")
                else:
                    print(f"[TOOL CALL] Result: {result_str}")
                
                # Intelligently limit result size for training efficiency
                if len(result_str) > 8000:
                    if isinstance(result, list) and len(result) > 20:
                        # For lists, return first 20 items with count info
                        truncated = result[:20]
                        result_str = json.dumps({
                            "results": truncated,
                            "total_count": len(result),
                            "truncated": True,
                            "message": f"Showing first 20 of {len(result)} results"
                        })
                    elif isinstance(result, dict):
                        # For dicts, preserve structure but limit values
                        result_str = json.dumps(result)[:8000] + "... [truncated]"
                    else:
                        # For other types, simple truncation
                        result_str = result_str[:8000] + "... [truncated]"
                
                self._instance_dict[instance_id]["last_result"] = result_str
                
                return result_str, 1.0, {"function": func_name, "success": True}
                
            except Exception as e:
                error_msg = f"Error: Exception in database function - {str(e)}"
                return error_msg, 0.0, {"error": True, "exception": str(e)}
        else:
            # Original GSM8K behavior
            answer = parameters.get("answer", "")
            if not isinstance(answer, str):
                answer = str(answer)

            if answer.startswith("#### "):
                self._instance_dict[instance_id]["response"] = answer
            else:
                self._instance_dict[instance_id]["response"] = "#### " + answer

            reward = await self.calc_reward(instance_id)
            # penalty for non improved answer submission
            tool_reward = 0.0 if reward > self._instance_dict[instance_id]["reward"] else -0.05
            # update the reward
            self._instance_dict[instance_id]["reward"] = reward

            return f"Current parsed {answer=} {reward=}", tool_reward, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        if self.database_type == "crmarena":
            # For CRMArena, reward is handled by the task evaluation
            return self._instance_dict[instance_id].get("reward", 0.0)
        else:
            # Original GSM8K behavior
            return gsm8k.compute_score(
                self._instance_dict[instance_id]["response"],
                self._instance_dict[instance_id]["ground_truth"],
                method="flexible",
                format_score=0.0,
                score=1.0,
            )

    async def release(self, instance_id: str, **kwargs) -> None:
        del self._instance_dict[instance_id]
