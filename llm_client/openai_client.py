from .client import LLMClient
from .config import RequestConfig
from typing import Tuple
import openai
from .measure.llm_metrics import LLMMetricTracker, LLMMetric
import json

class OpenAIClient(LLMClient):
    def __init__(self, base_url: str, api_key: str):
        super().__init__()
        self.__internal_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.llm_metric_tracker = LLMMetricTracker()
    
    async def request_llm(self, request_config: RequestConfig) -> Tuple[str, LLMMetric]:
        """Sends a request to the LLM and tracks metrics."""
        self.llm_metric_tracker.reset()
        message = ""

        # Initialize the tracker
        self.llm_metric_tracker.inititialize()

        # Start the streaming request
        async with self.__internal_client.chat.completions.with_streaming_response.create(
            model=request_config.model,
            messages=[
                {"role": "system", "content": request_config.system_prompt},
                {"role": "user", "content": request_config.user_prompt},
            ],
            stream=True,  # Ensure streaming is enabled
        ) as response:
            async for line in response.iter_lines():
                if line:
                    if line.startswith('data: '):
                        event_data = line[6:]  # Remove 'data: ' prefix
                        if event_data == '[DONE]':
                            break
                        try:
                            chunk_data = json.loads(event_data)
                            content = chunk_data['choices'][0]['delta']['content']
                            message += content
                            self.llm_metric_tracker.token_generated()
                        except (json.JSONDecodeError, KeyError):
                            continue

        # Return the final message and metrics
        return message, LLMMetric(
            time_to_first_token=self.llm_metric_tracker.time_to_first_token,
            end_to_end_token_per_second=self.llm_metric_tracker.token_per_second,
            tokens_generated=self.llm_metric_tracker.count_tokens_generated,
        )