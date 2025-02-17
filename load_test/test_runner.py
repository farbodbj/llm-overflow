from llm_client import LLMClient, RequestConfig
from test_utils.prompt_gen import RandomPromptGen
import asyncio
from typing import List, Dict
import time
import datetime
import json
from dataclasses import dataclass
@dataclass
class EvalMetrics:
    response_time: float
    generated_tokens: int
    tokens_per_second: float
    
class LoadTester:
    def __init__(self, client: LLMClient, model: str, corpus: List[str], max_users: int = 10, step: int = 1, repetition_count: int = 1):
        self.client = client
        self.prompt_gen = RandomPromptGen(corpus)
        self.max_users = max_users
        self.repetition_count = repetition_count
        self.step = step
        self.results: List[Dict] = []
        self.model = model

    async def _simulate_user(self, user_id: int) -> EvalMetrics:
        """Simulates a single user making a request to the LLM."""
        try:
            
            prompt = self.prompt_gen.get_prompt()
            request_config = RequestConfig(
                model=self.model,
                system_prompt="",
                user_prompt=prompt,
            )
        
            start_time = time.monotonic()
            _, metrics = await self.client.request_llm(request_config)
            end_time = time.monotonic()

        
            response_time = end_time - start_time
            tokens_per_second = metrics.tokens_generated / response_time if response_time > 0 else 0

            return EvalMetrics(
                response_time = response_time,
                generated_tokens = metrics.tokens_generated,
                tokens_per_second = tokens_per_second
            )
        except Exception as e:
            print(f"User {user_id} encountered an error: {e}")
            return None

    async def run_test(self):
        """Runs the load test with increasing numbers of concurrent users."""
        for num_users in range(1, self.max_users + 1, self.step):
            print(f"Testing with {num_users} concurrent users...")

            for _ in range(self.repetition_count):
                tasks = [self._simulate_user(i) for i in range(num_users)]
                results = [result for result in await asyncio.gather(*tasks) if result != None]

            total_tokens = sum(result.generated_tokens for result in results)
            total_time = max(result.response_time for result in results) 
            successful_requests = sum(1 for result in results if result.response_time != -1)
            avg_tokens_per_second = total_tokens / total_time if total_time > 0 else 0

        
            self.results.append({
                "num_users": num_users,
                "total_tokens": total_tokens,
                "total_time": total_time,
                "successful_requests": successful_requests,
                "avg_tokens_per_second": avg_tokens_per_second,
            })

            self.report_results(num_users, successful_requests, total_tokens, total_time, avg_tokens_per_second)

    def report_results(self, num_users, successful_requests, total_tokens, total_time, avg_tokens_per_second):
        """Reports the test results in a formatted manner."""
        print(f"Results for {num_users} users:")
        print(f"  Successful Requests: {successful_requests}/{num_users}")
        print(f"  Total Tokens: {total_tokens}")
        print(f"  Total Time: {total_time:.2f} seconds")
        print(f"  Avg Tokens per Second: {avg_tokens_per_second:.2f}", end='\n\n')
        
    def get_results(self) -> List[Dict]:
        """Returns the collected load test results."""
        return self.results
    
    def save_results(self, path: str = f'benchmark_{datetime.datetime.now().strftime("%m_%d_%Y_%H_%M")}.json'):
        with open(path, 'x') as f:
            json.dump(self.get_results(), f, indent=4)