from typing import Dict, Optional, Literal, Any
import os
import json
import sys
from abc import ABC, abstractmethod
from litellm import completion

# Import your custom Azure LLM system
try:
    # Add the inference directory to path to import your LLM system
    inference_path = os.path.join(os.path.dirname(__file__), '..', '..', 'inference')
    if inference_path not in sys.path:
        sys.path.append(inference_path)
    
    from llm_api import get_client, get_llm_response_normal
    AZURE_LLM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Azure LLM system: {e}")
    AZURE_LLM_AVAILABLE = False

class BaseLLMController(ABC):
    @abstractmethod
    def get_completion(self, prompt: str) -> str:
        """Get completion from LLM"""
        pass

class AzureLLMController(BaseLLMController):
    """Controller using your custom Azure LLM system"""
    def __init__(self, model: str = "deepprompt-gpt-5-2025-08-07-global"):
        if not AZURE_LLM_AVAILABLE:
            raise ImportError("Azure LLM system not available. Check your inference/llm_api.py file.")
        self.model = model
        # Create a simple opt object that mimics your system
        from types import SimpleNamespace
        self.opt = SimpleNamespace(deployment_name=model)
    
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        try:
            # Use your custom Azure system
            messages = [
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ]
            
            # Get response using your system
            response_content, usage = get_llm_response_normal(messages, self.opt)
            return response_content
            
        except Exception as e:
            print(f"Error with Azure LLM: {e}")
            # Fallback to empty response if format is specified
            if response_format and "json_schema" in response_format:
                return self._generate_empty_response(response_format)
            return "{}"
    
    def _generate_empty_response(self, response_format: dict) -> str:
        """Generate empty response based on JSON schema"""
        if "json_schema" not in response_format:
            return "{}"
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return json.dumps(result)
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

class OpenAIController(BaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            self.model = model
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            if api_key is None:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
    
    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_format,
            temperature=temperature,
            max_tokens=1000
        )
        return response.choices[0].message.content

class OllamaController(BaseLLMController):
    def __init__(self, model: str = "llama2"):
        from ollama import chat
        self.model = model
    
    def _generate_empty_value(self, schema_type: str, schema_items: dict = None) -> Any:
        if schema_type == "array":
            return []
        elif schema_type == "string":
            return ""
        elif schema_type == "object":
            return {}
        elif schema_type == "number":
            return 0
        elif schema_type == "boolean":
            return False
        return None

    def _generate_empty_response(self, response_format: dict) -> dict:
        if "json_schema" not in response_format:
            return {}
            
        schema = response_format["json_schema"]["schema"]
        result = {}
        
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                result[prop_name] = self._generate_empty_value(prop_schema["type"], 
                                                            prop_schema.get("items"))
        
        return result

    def get_completion(self, prompt: str, response_format: dict, temperature: float = 0.7) -> str:
        try:
            response = completion(
                model="ollama_chat/{}".format(self.model),
                messages=[
                    {"role": "system", "content": "You must respond with a JSON object."},
                    {"role": "user", "content": prompt}
                ],
                response_format=response_format,
            )
            return response.choices[0].message.content
        except Exception as e:
            empty_response = self._generate_empty_response(response_format)
            return json.dumps(empty_response)

class LLMController:
    """LLM-based controller for memory metadata generation"""
    def __init__(self, 
                 backend: Literal["openai", "ollama", "azure"] = "azure",
                 model: str = "deepprompt-gpt-5-2025-08-07-global", 
                 api_key: Optional[str] = None):
        if backend == "azure":
            self.llm = AzureLLMController(model)
        elif backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        else:
            raise ValueError("Backend must be one of: 'azure', 'openai', 'ollama'")
            
    def get_completion(self, prompt: str, response_format: dict = None, temperature: float = 0.7) -> str:
        return self.llm.get_completion(prompt, response_format, temperature)
