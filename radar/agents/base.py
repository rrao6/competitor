"""
Base agent class for Tubi Radar agents.

Provides common functionality for all agents including LLM configuration
and tool access patterns.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from radar.config import get_config, get_settings


class BaseAgent(ABC):
    """
    Abstract base class for all Radar agents.
    
    Each agent has:
    - A system prompt defining its role
    - Access to a specific subset of tools
    - Configuration for the LLM model
    """
    
    # Override in subclasses
    agent_role: str = "base_agent"
    system_prompt: str = "You are a helpful assistant."
    
    def __init__(
        self,
        model_override: Optional[str] = None,
        temperature_override: Optional[float] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            model_override: Override the default model
            temperature_override: Override the default temperature
        """
        self.config = get_config()
        self.settings = get_settings()
        self._model_override = model_override
        self._temperature_override = temperature_override
        self._llm: Optional[ChatOpenAI] = None
    
    @property
    def model_name(self) -> str:
        """Get the model name for this agent."""
        if self._model_override:
            return self._model_override
        return self.config.global_config.models.structured
    
    @property
    def temperature(self) -> float:
        """Get the temperature for this agent."""
        if self._temperature_override is not None:
            return self._temperature_override
        return self.config.global_config.temperature.classification
    
    def get_llm(self, for_structured_output: bool = False) -> ChatOpenAI:
        """
        Get the LLM instance for this agent.
        
        Args:
            for_structured_output: If True, use model optimized for structured outputs
        
        Returns:
            Configured ChatOpenAI instance
        """
        if self._llm is None or for_structured_output:
            model = self.model_name
            if for_structured_output:
                model = self.config.global_config.models.structured
            
            self._llm = ChatOpenAI(
                model=model,
                temperature=self.temperature,
                api_key=self.settings.openai_api_key,
            )
        
        return self._llm
    
    def get_structured_llm(self, schema: type[BaseModel]) -> ChatOpenAI:
        """
        Get an LLM configured for structured output with a specific schema.
        
        Args:
            schema: Pydantic model class for the expected output
        
        Returns:
            ChatOpenAI instance with structured output
        """
        llm = self.get_llm(for_structured_output=True)
        return llm.with_structured_output(schema)
    
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """
        Execute the agent's main task.
        
        Override in subclasses to implement agent-specific logic.
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(role={self.agent_role}, model={self.model_name})>"

