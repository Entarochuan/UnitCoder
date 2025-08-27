import os
from pathlib import Path
import json
from typing import Set, Generator, Dict
import asyncio
import argparse
from UnitCoder_pipeline import CodeTestingPipeline, PipelineConfig, ModelConfig

def get_processed_ids(save_dir: str) -> Set[str]:
    """get processed ids"""
    processed_ids = set()
    save_dir_path = Path(save_dir)
    
    # check passed.jsonl
    passed_path = save_dir_path / "passed.jsonl"
    if passed_path.exists():
        with open(passed_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                processed_ids.add(item['id'])
                
    # check failed.jsonl
    failed_path = save_dir_path / "failed.jsonl"
    if failed_path.exists():
        with open(failed_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                processed_ids.add(item['id'])
                
    return processed_ids

def load_data_generator(input_path: str, processed_ids: Set[str]) -> Generator[Dict, None, None]:
    """load data generator"""
    if os.path.isdir(input_path):
        # if input is a directory, traverse all .jsonl files
        for filename in sorted(os.listdir(input_path)):
            if filename.endswith('.jsonl'):
                file_path = os.path.join(input_path, filename)
                yield from process_single_file(file_path, processed_ids)
    else:
        # if input is a single file
        yield from process_single_file(input_path, processed_ids)

def process_single_file(file_path: str, processed_ids: Set[str]) -> Generator[Dict, None, None]:
    """process single file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if item['id'] not in processed_ids:
                yield item

async def main():
    parser = argparse.ArgumentParser()
    
    # basic parameters
    parser.add_argument('--input_path', required=True, help='Input file or directory path')
    parser.add_argument('--save_debug_dir', required=True, help='Directory to save results')
    parser.add_argument('--save_polish_dir', required=True, help='Directory to save results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--max_debug_rounds', type=int, default=3, help='Max debug rounds')
    parser.add_argument('--max_pass_per_round', type=int, default=2, help='Max pass per round')
    
    # concurrency control parameters
    parser.add_argument('--unittest_workers', type=int, default=32, help='Number of API workers')
    parser.add_argument('--debug_workers', type=int, default=16, help='Number of code execution workers')
    parser.add_argument('--polish_workers', type=int, default=16, help='Number of polish workers')
    parser.add_argument('--code_workers', type=int, default=16, help='Number of code execution workers')
    
    # unit test model parameters
    parser.add_argument('--unittest_api_base', required=True, help='Unittest model API base URL')
    parser.add_argument('--unittest_model', default="Meta-Llama-3-70B-Instruct", help='Model name for unittest generation')
    parser.add_argument('--unittest_prompt', 
                       default="You are a program testing expert, please help me write some test cases based on the corresponding function.",
                       help='System prompt for unittest generation')
    parser.add_argument('--unittest_temperature', type=float, default=0.2, help='Temperature for unittest generation')
    parser.add_argument('--unittest_max_tokens', type=int, default=2048, help='Max tokens for unittest generation')

    # debug model parameters
    parser.add_argument('--debug_api_base', required=True, help='Debug model API base URL')
    parser.add_argument('--debug_model', default="Meta-Llama-3-70B-Instruct", help='Model name for code debugging')
    parser.add_argument('--debug_prompt', 
                       default="You are a coding expert.",
                       help='System prompt for code debugging')
    parser.add_argument('--debug_temperature', type=float, default=0.5, help='Temperature for code debugging')
    parser.add_argument('--debug_max_tokens', type=int, default=2048, help='Max tokens for code debugging')
    
    # polish model parameters
    
    parser.add_argument('--polish_api_base', required=True, help='Polish model API base URL')
    parser.add_argument('--polish_model', default="Meta-Llama-3-70B-Instruct", help='Model name for code polishing')
    parser.add_argument('--polish_prompt', 
                       default="You are a coding expert.",
                       help='System prompt for code polishing')
    parser.add_argument('--polish_temperature', type=float, default=0.5, help='Temperature for code polishing')
    parser.add_argument('--polish_max_tokens', type=int, default=2048, help='Max tokens for code polishing')
    
    args = parser.parse_args()

    # config models
    unittest_config = ModelConfig(
        api_key="EMPTY",
        base_url=args.unittest_api_base,
        model_name=args.unittest_model,
        system_prompt=args.unittest_prompt,
        temperature=args.unittest_temperature,
        max_tokens=args.unittest_max_tokens
    )

    debug_config = ModelConfig(
        api_key="EMPTY",
        base_url=args.debug_api_base,
        model_name=args.debug_model,
        system_prompt=args.debug_prompt,
        temperature=args.debug_temperature,
        max_tokens=args.debug_max_tokens
    )

    polish_config = ModelConfig(
        api_key="EMPTY",
        base_url=args.polish_api_base,
        model_name=args.polish_model,
        system_prompt=args.polish_prompt,
        temperature=args.polish_temperature,
        max_tokens=args.polish_max_tokens
    )

    # config pipeline
    pipeline_config = PipelineConfig(
        unittest_config=unittest_config,
        debug_config=debug_config,
        polish_config=polish_config,
        save_debug_dir=args.save_debug_dir,
        save_polish_dir=args.save_polish_dir,
        unittest_workers=args.unittest_workers,
        debug_workers=args.debug_workers,
        polish_workers=args.polish_workers,
        code_workers=args.code_workers,
        batch_size=args.batch_size,
        max_debug_rounds=args.max_debug_rounds,
        max_pass_per_round=args.max_pass_per_round
    )

    # create pipeline instance
    pipeline = CodeTestingPipeline(pipeline_config)
    
    os.makedirs(args.save_debug_dir, exist_ok=True)

    # get processed ids
    processed_ids = get_processed_ids(args.save_debug_dir)
    data_generator = load_data_generator(args.input_path, processed_ids)
    
    # process in batches
    while True:
        batch = []
        for _ in range(args.batch_size):
            try:
                item = next(data_generator)
                batch.append(item)
            except StopIteration:
                break
        
        if not batch:
            break
            
        await pipeline.process_batch(batch)

if __name__ == "__main__":
    asyncio.run(main()) 