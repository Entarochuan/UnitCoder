#!/bin/bash

# 1. MUST SET: place the execution directory in an isolated directory to avoid the dangerous code!!!!

EXEC_DIR=""
export EXEC_DIR

# 2. Config for input and output
INPUT_PATH="./data/demo/stage1_filtering"  # input data directory path
SAVE_DEBUG_DIR="./data/demo/stage2_iterative_fix"  # result save directory
SAVE_POLISH_DIR="./data/demo/stage3_polish"  # result save directory

# 3. Config for API call
UNITTEST_API="http://10.130.128.117:21001/v1"   # API base for unit test generation model
UNITTEST_API_KEY="EMPTY"
DEBUG_API="http://10.130.128.117:21001/v1"      # API base for code debugging model
DEBUG_API_KEY="EMPTY"
POLISH_API="http://10.130.128.117:21001/v1"      # API base for code polishing model
POLISH_API_KEY="EMPTY"

UNITTEST_WORKERS=32    # API workers for unit test generation model
DEBUG_WORKERS=16   # API workers for code debugging model
POLISH_WORKERS=16   # API workers for code polishing model
CODE_WORKERS=64   # workers for code execution
BATCH_SIZE=16     # data batch size 

# 4. Config for unit test generation model
UNITTEST_TEMP=0.2 # temperature 
UNITTEST_MAX_TOKENS=2048 # max tokens
UNITTEST_MODEL="qwen3.2-235b-instruct-2507" # model name
UNITTEST_PROMPT="""You are a professional software testing expert. Your task is to write comprehensive unit tests for the given function.

Please follow these guidelines:
1. Write tests that cover different scenarios including:
   - Normal/expected inputs
   - Edge cases
   - Invalid/unexpected inputs
   - Boundary conditions

2. Each test case should:
   - Class name should be TestCases
   - Description should be concise and clear
   - Include assertions that verify both return values and expected behavior
   - Be independent of other test cases, and should be self-contained
   - Include brief comments explaining the test purpose

3. Test structure requirements:
   - Use the unittest framework
   - Create a proper test class inheriting from unittest.TestCase
   - Include setUp/tearDown methods if necessary

4. Important:
   - Only output the test code within Python code blocks
   - Ensure all necessary imports are included
   - Focus on functionality testing rather than implementation details
   - Write tests that are maintainable and readable

Please analyze the given function and generate appropriate unit tests following these guidelines.
Your output format should be like this:
\`\`\`python
import unittest
class TestCases(unittest.TestCase):
    def test_case_1(self):
        # Test purpose: Verify the function handles normal inputs correctly
        self.assertEqual(function_name(input1, input2), expected_output1)
    def test_case_2(self):
        ... 
\`\`\`
Do not modify the class name(TestCases).
"""

# 5. Config for code debugging model
DEBUG_TEMP=0.5 # temperature 
DEBUG_MAX_TOKENS=2048 # max tokens
MAX_DEBUG_ROUNDS=3  # max debug rounds
MAX_PASS_PER_ROUND=2  # max pass per round(how many pass examples are kept in each round)
DEBUG_MODEL="qwen3.2-235b-instruct-2507" # model name
DEBUG_PROMPT="""You are a powerful coding expert specialized in code debugging and optimization. Your task is to fix the given code based on unit test results and error messages.

Please follow these guidelines:
1. Carefully analyze:
   - The original code implementation
   - Failed test cases and their error messages
   - Test requirements and expected behavior

2. When fixing the code:
   - Make minimal necessary changes to fix the issues
   - Maintain the original code structure when possible
   - Ensure the solution is efficient and clean

3. Important:
   - Only output the fixed code within Python code blocks
   - Ensure the solution passes all test cases
   - Focus on addressing the specific test failures
   - Maintain code readability and best practices

Please analyze the code and test failures, then provide the corrected implementation.
Your output format should be like this:
\`\`\`python
# imports
def function_name(params):
    # Fixed implementation
    ...
\`\`\`
"""

# 6. Config for code polishing model
POLISH_TEMP=0.5 # temperature 
POLISH_MAX_TOKENS=2048 # max tokens
POLISH_MODEL="qwen3.2-235b-instruct-2507" # model name
POLISH_PROMPT="""You are a coding expert specialized in code documentation, code generation and optimization. Your task consists of two parts:

**Part 1: Function Documentation**
Generate comprehensive documentation for the function including:
- Function description and purpose
- Parameters with types and descriptions
- Return value with type and description
- Requirements (necessary imports and dependencies)
- Raises section (if applicable)
- Usage examples with input/output

**Part 2: Code Generation with Inline Comments**
Generate the actual code implementation with:
- Clear inline comments explaining key logic and important parts
- Proper error handling where needed
- Clean, readable code structure
- All necessary imports
- Code that passes the given unit tests
- Maintain the original function name

Ensure both parts are well-structured, accurate, and follow Python best practices. Do not change the original code functionality.
"""

# 6. Execution

# create result save directory
mkdir -p ${SAVE_DEBUG_DIR}
mkdir -p ${SAVE_POLISH_DIR}

# 记录运行时间
echo "Pipeline started at $(date)"

# 运行pipeline
python run_pipeline.py \
    --input_path ${INPUT_PATH} \
    --save_debug_dir ${SAVE_DEBUG_DIR} \
    --save_polish_dir ${SAVE_POLISH_DIR} \
    --batch_size ${BATCH_SIZE} \
    --unittest_workers ${UNITTEST_WORKERS} \
    --debug_workers ${DEBUG_WORKERS} \
    --polish_workers ${POLISH_WORKERS} \
    --code_workers ${CODE_WORKERS} \
    --max_debug_rounds ${MAX_DEBUG_ROUNDS} \
    --max_pass_per_round ${MAX_PASS_PER_ROUND} \
    --unittest_api_base ${UNITTEST_API} \
    --unittest_model ${UNITTEST_MODEL} \
    --unittest_prompt "${UNITTEST_PROMPT}" \
    --unittest_temperature ${UNITTEST_TEMP} \
    --unittest_max_tokens ${UNITTEST_MAX_TOKENS} \
    \
    --debug_api_base ${DEBUG_API} \
    --debug_model ${DEBUG_MODEL} \
    --debug_prompt "${DEBUG_PROMPT}" \
    --debug_temperature ${DEBUG_TEMP} \
    --debug_max_tokens ${DEBUG_MAX_TOKENS} \
    \
    --polish_api_base ${POLISH_API} \
    --polish_model ${POLISH_MODEL} \
    --polish_prompt "${POLISH_PROMPT}" \
    --polish_temperature ${POLISH_TEMP} \
    --polish_max_tokens ${POLISH_MAX_TOKENS} \
    2>&1 | tee ${SAVE_DEBUG_DIR}/pipeline.log

echo "Pipeline finished at $(date)" 