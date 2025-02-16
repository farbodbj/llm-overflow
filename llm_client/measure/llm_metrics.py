from dataclasses import dataclass
import time

class LLMMetricTracker:
    def __init__(self):
        self._token_generated_ts = []
        self._init_time = None
        
    @property
    def time_to_first_token(self)->float:
        return (self._token_generated_ts[0] - self._init_time)
    
    @property
    def token_per_second(self)->float:
        return len(self._token_generated_ts) / (self._token_generated_ts[-1] - self._init_time)
        
    @property
    def count_tokens_generated(self)->int:
        return len(self._token_generated_ts)    
    
    def inititialize(self):
        if not self._init_time:
            self._init_time = time.monotonic()
        
    def token_generated(self):
        self._token_generated_ts.append(time.monotonic())
    
    def reset(self):
        self._token_generated_ts = []
        self._init_time = None

@dataclass
class LLMMetric:
    time_to_first_token: int
    end_to_end_token_per_second: float
    tokens_generated: int
    