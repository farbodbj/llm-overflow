import abc
from .config import RequestConfig
from typing import Tuple
from .measure.llm_metrics import LLMMetric

class LLMClient(abc.ABC):
    @abc.abstractmethod
    def request_llm(self, request_config: RequestConfig)-> Tuple[str,LLMMetric]:
        pass       