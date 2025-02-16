from llm_client import OpenAIClient
from test_utils.prompt_gen import CorpusReader
from load_test import LoadTester
import asyncio
import argparse

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Load test an LLM API.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL of the LLM API.")
    parser.add_argument("--api-key", type=str, required=True, help="API key for the LLM API.")
    parser.add_argument("--max-users", type=int, default=10, help="Maximum number of concurrent users to simulate.")
    parser.add_argument("--step", type=int, default=1, help="Step size for increasing the number of users.")
    parser.add_argument("--save-results", type=str, help="Path to save the results as a JSON file.")
    parser.add_argument("--custom-corpus-path", default='pg1661.txt', type=str, help="Path to custom corpus .txt file to create random prompts from")
    return parser.parse_args()

async def main():
    args = parse_args()
    
    client = OpenAIClient(args.base_url, api_key=args.api_key)
    corpus = CorpusReader().read(args.custom_corpus_path)
    load_tester = LoadTester(client, args.model, corpus, max_users=args.max_users, step=args.step)

    await load_tester.run_test()
    if args.save_results:
        load_tester.save_results()
    else:
        print(load_tester.get_results())

asyncio.run(main())
