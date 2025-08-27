import json
import re
import os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from openai import AsyncOpenAI
import random
from tqdm import tqdm
import asyncio
import sys
import logging
import aiofiles
import traceback
from dataclasses import dataclass
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "secure_sandbox"))
from secure_sandbox.examine import untrusted_check

@dataclass
class ModelConfig:
    """模型配置"""
    api_key: str
    base_url: str
    model_name: str
    system_prompt: str
    temperature: float = 0.0
    max_tokens: int = 2048
    
@dataclass 
class PipelineConfig:
    """Pipeline Config"""
    unittest_config: ModelConfig
    debug_config: ModelConfig
    polish_config: ModelConfig
    save_debug_dir: str
    save_polish_dir: str
    unittest_workers: int = 32  # UnitTest API并发数
    debug_workers: int = 16  # Debug API并发数
    polish_workers: int = 16  # Polish API并发数
    code_workers: int = 16  # Code执行并发数
    max_debug_rounds: int = 3  # 最大调试轮数
    max_pass_per_round: int = 2  # 每轮保留通过样例数
    batch_size: int = 64  # 批处理大小
    save_debug_dir: str = ""  # 调试结果保存目录
    save_polish_dir: str = ""  # polish结果保存目录

class CodeTestingPipeline:
    def __init__(self, config: PipelineConfig):
        """基础的参数"""
        self.config = config
        self.unittest_client = AsyncOpenAI(
            api_key=config.unittest_config.api_key,
            base_url=config.unittest_config.base_url
        )
        self.debug_client = AsyncOpenAI(
            api_key=config.debug_config.api_key,
            base_url=config.debug_config.base_url
        )
        self.polish_client = AsyncOpenAI(
            api_key=config.polish_config.api_key,
            base_url=config.polish_config.base_url
        )
        
        # 并发相关控制
        self.sem_unittest_api = asyncio.Semaphore(config.unittest_workers)
        self.sem_debug_api = asyncio.Semaphore(config.debug_workers)
        self.sem_polish_api = asyncio.Semaphore(config.polish_workers)
        self.sem_code = asyncio.Semaphore(config.code_workers)
        
        # os.makedirs(config.save_dir, exist_ok=True)
        if config.save_debug_dir:
            os.makedirs(config.save_debug_dir, exist_ok=True)
        if config.save_polish_dir:
            os.makedirs(config.save_polish_dir, exist_ok=True)
        self.logger = self._setup_logger()
        
        # 添加已处理ID的记录
        self.processed_count = 0
        
    def _setup_logger(self) -> logging.Logger:
        """Logger 配置"""
        logger = logging.getLogger("CodeTestingPipeline")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                )
            )
            logger.addHandler(handler)
            
        return logger

    async def generate_unittest(self, item: Dict) -> Dict:
        """生成单元测试"""
        try:
            prefix = "\n".join(item.get('import_code', [])) + "\n" if item.get('import_code') else ""
            prompt = self._get_unittest_prompt(prefix + item['content'])
            
            async with self.sem_unittest_api:
                reply = await self._api_call(
                    self.unittest_client,
                    prompt,
                    self.config.unittest_config
                )
                
            test_code = self._extract_python_code(reply.choices[0].message.content)
            if not test_code:
                raise ValueError("No valid test code generated")
            
            # 执行初始代码测试    
            initial_status = await self._execute_code(
                prefix + item['content'],
                test_code
            )
            
            # 保存中间结果
            # intermediate_result = {
            #     "id": item["id"],
            #     "original_code": prefix + item["content"],
            #     "test_code": test_code,
            #     "execution_status": initial_status,
            #     "timestamp": datetime.now().isoformat()
            # }
            
            # await self._save_intermediate_results(intermediate_result)
            
            return {
                "id": item["id"],
                "code": prefix + item["content"],
                "test_code": test_code,
                "import_code": item.get("import_code", [])
            }
            
        except Exception as e:
            self.logger.error(f"Error generating unittest for item {item['id']}: {str(e)}")
            raise

    async def debug_code(self, item: Dict, round: int) -> List[Dict]:
        """调试代码"""
        try:
            prompt = self._get_debug_prompt(item)
            
            async with self.sem_debug_api:
                replies = await self._api_call(
                    self.debug_client,
                    prompt,
                    self.config.debug_config,
                    n=3
                )
                
            exec_results = []
            return_results = []
            for reply in replies.choices:
                code = self._extract_python_code(reply.message.content)
                if code and self._is_safe_code(code):
                    status = await self._execute_code(code, item["original"]["test_code"])
                    exec_info = {
                        "code": code,
                        "test_code": item["original"]["test_code"],
                        "status": status,
                        "round": round
                    }
                    exec_results.append(exec_info)
                    if status[0] == "pass":
                        return_results.append(exec_info) 
            
            # 如果达到最大轮数且未通过，返回执行结果
            if not any(r["status"][0] == "pass" for r in exec_results):
                return exec_results[:1] # 返回最后一个执行结果
            else:
                return return_results[:self.config.max_pass_per_round]
            
        except Exception as e:
            self.logger.error(f"Error debugging code for item {item['id']}: {str(e)}")
            return []

    async def polish_code(self, item: Dict) -> Optional[Dict]:
        """对通过的代码进行polish优化"""
        try:
            # 获取通过的代码和测试
            if item['debug_rounds']:
                # 从debug结果中获取通过的代码
                for round_results in item['debug_rounds']:
                    for result in round_results:
                        if result["status"][0] == "pass":
                            code = result["code"]
                            test_code = result["test_code"]
                            break
                    else:
                        continue
                    break
                else:
                    # 如果没有找到通过的代码，使用原始代码
                    code = item['original']['code']
                    test_code = item['original']['test_code']
            else:
                # 原始代码就通过了
                code = item['original']['code']
                test_code = item['original']['test_code']
            
            prompt = self._get_polish_prompt(code, test_code)
            
            async with self.sem_polish_api:
                reply = await self._api_call(
                    self.polish_client,
                    prompt,
                    self.config.polish_config
                )
                
            polished_code = self._extract_python_code(reply.choices[0].message.content)
            if not polished_code:
                self.logger.warning(f"No valid polished code generated for item {item['id']}")
                return None
            
            # 测试polish后的代码
            polish_status = await self._execute_code(polished_code, test_code)
            
            if polish_status[0] == "pass":
                self.logger.info(f"Polished code passed tests for item {item['id']}")
                return {
                    "code": polished_code,
                    "test_code": test_code,
                    "status": polish_status,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                self.logger.warning(f"Polished code failed tests for item {item['id']}, keeping original")
                return None
                
        except Exception as e:
            self.logger.error(f"Error polishing code for item {item['id']}: {str(e)}")
            return None

    async def process_item(self, item: Dict) -> Dict:
        """处理单个样例的完整流程"""
        try:
            self.logger.info(f"Processing item {item['id']}")
            
            # 生成单元测试
            test_result = await self.generate_unittest(item)
            
            # 验证初始代码
            initial_status = await self._execute_code(
                test_result["code"],
                test_result["test_code"]
            )
            
            results = {
                "id": item["id"],
                "original": {
                    "code": test_result["code"],
                    "test_code": test_result["test_code"],
                    "status": initial_status
                },
                "debug_rounds": [], 
                "polished": False, 
                "polished_result": None, 
            }
            
            # 如果初始代码未通过,进行多轮调试
            passed = False
            if initial_status[0] != "pass":
                for round in range(self.config.max_debug_rounds):
                    self.logger.info(f"Starting debug round {round+1} for item {item['id']}")
                    debug_results = await self.debug_code(results, round)
                    if debug_results:
                        results["debug_rounds"].append(debug_results)
                        if any(r["status"][0] == "pass" for r in debug_results):
                            self.logger.info(f"Found passing solution in round {round+1}")
                            passed = True
                            break
            else:
                passed = True
            
            # 对通过的代码进行polish
            if passed:
                self.logger.info(f"Starting polish for item {item['id']}")
                polish_result = await self.polish_code(results)
                if polish_result:
                    results["polish"] = polish_result
                    results["polished"] = True
                
            await self._save_results(results, passed)
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing item {item['id']}: {str(e)}\n{traceback.format_exc()}")
            return {"id": item["id"], "error": str(e)}

    async def process_batch(self, items: List[Dict]):
        """批量处理样例"""
        self.logger.info(f"Processing batch of {len(items)} items")
        pbar = tqdm(
            total=len(items),
            initial=self.processed_count,
            desc="Processing items"
        )
        
        tasks = []
        for item in items:
            task = asyncio.create_task(self.process_item(item))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if not isinstance(result, Exception):
                self.processed_count += 1
                pbar.update(1)
        
        pbar.close()

    async def _api_call(
        self, 
        client: AsyncOpenAI, 
        prompt: str, 
        config: ModelConfig,
        **kwargs
    ) -> Any:
        """封装API调用"""
        return await client.chat.completions.create(
            model=config.model_name,
            messages=[
                {"role": "system", "content": config.system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            **kwargs
        )

    async def _execute_code(self, code: str, test_code: str) -> Tuple[str, str]:
        """执行代码"""
        async with self.sem_code:
            status = await asyncio.to_thread(
                untrusted_check,
                code,
                test_code,
                "check_funcs",
                max_as_limit=512*1024,
                max_data_limit=64*1024,
                max_stack_limit=10,
                min_time_limit=10,
                gt_time_limit=10
            )
        return status

    async def _save_results(self, results: Dict, passed: bool):
        """保存结果到jsonl文件"""
        if passed:
            save_path = Path(self.config.save_debug_dir) / "passed.jsonl"
        else:
            save_path = Path(self.config.save_debug_dir) / "failed.jsonl"
            
        async with aiofiles.open(save_path, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(results, ensure_ascii=False) + '\n')
        
        # 如果有polish结果，保存到polish目录
        if passed and "polished" in results and results["polished"]:
            polish_save_path = Path(self.config.save_polish_dir) / "polished.jsonl"
            async with aiofiles.open(polish_save_path, 'a', encoding='utf-8') as f:
                # 保存完整的polish结果
                await f.write(json.dumps(results, ensure_ascii=False) + '\n')

    def _is_safe_code(self, code: str) -> bool:
        """检查代码安全性"""
        dangerous_commands = [
            "rm -rf", "os.killpg", "os.kill", "getpwd",
            "kill", "getpid", "getpgid", "input(", "sys.argv",
            "os.remove", "os.rmdir", "rmtree", "os.system"
        ] # 可以添加一些其他危险命令，确保pipeline的安全性。
        
        return not any(cmd in code for cmd in dangerous_commands)

    def _extract_python_code(self, text: str) -> Optional[str]:
        """从文本中提取Python代码"""
        pattern = r"```python\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _get_unittest_prompt(self, code: str) -> str:
        """生成单元测试提示"""
        return f"The function is:\n```python\n{code}\n```"

    def _get_debug_prompt(self, item: Dict) -> str:
        """生成调试提示"""
        if len(item['debug_rounds']) == 0:
            code = item['original']['code']
            test_code = item['original']['test_code']
            status = item['original']['status']
        else:
            code = item['debug_rounds'][-1][0]['code']
            test_code = item['original']['test_code']
            status = item['debug_rounds'][-1][0]['status']
        
        return f"""Given the following code, unittest and failed test case output, please modify the code to pass the unittest. The modified code should contain only one function.

The code is: 
```python
{code}
```

The unittest is: 
```python
{test_code}
```

The output is: 
{status[1]}

The modified code is: 
""" 

    def _get_polish_prompt(self, code: str, test_code: str) -> str:
        """生成polish提示"""
        return f"""Given the following code and unit test, please polish and optimize the code while maintaining its functionality.

The code should include:
1. Comprehensive function documentation (description, parameters, returns, requirements, raises if applicable, examples)
2. Clear inline comments explaining key logic and important parts
3. Proper error handling where needed
4. Clean, readable code structure
5. All necessary imports

The code is:
```python
{code}
```

The unit test is:
```python
{test_code}
```

Please provide the polished code with documentation and inline comments:
"""

    async def _save_intermediate_results(self, result: Dict):
        """保存中间结果到jsonl文件"""
        intermediate_path = Path(self.config.save_debug_dir) / "intermediate_results.jsonl"
        async with aiofiles.open(intermediate_path, 'a', encoding='utf-8') as f:
            await f.write(json.dumps(result, ensure_ascii=False) + '\n')