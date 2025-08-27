import itertools
import multiprocessing
import os
import sys
import time
import types
import unittest
from multiprocessing import Array, Value, Manager
from typing import Any, Dict, List, Tuple, Union
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# print(os.path.dirname(os.path.abspath(__file__)))

from exam_utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
    safe_environment,
    TIMEOUT_LIMIT,
)

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_SUCCESS = 0
_FAILED = 1
_TIMEOUT = 2
_UNKNOWN = 3
_mapping = {_SUCCESS: PASS, _FAILED: FAIL, _TIMEOUT: TIMEOUT, _UNKNOWN: None}

def unsafe_execute(
    entry_point: str,
    code: str,
    test_code: str,
    timeout: float,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float,
    stat,  # Value
    details,  # Array
    dtype
):
    
    # import os 
    # test_code_dir = os.path.join(save_dir, "test_codes")
    # os.makedirs(test_code_dir, exist_ok=True)
    # test_file_path = os.path.join(test_code_dir, f"{os.getpid()}_test.py")
    # with open(test_file_path, 'w') as test_file:
    #     test_file.write(code + "\n\n" + "################## Test cases are as folows ##################\n\n"+ test_code)
    
    # if "rmtree" in code or "rmtree" in test_code : 
    #     details["ALL"] = "Trying to remove dictionary, skip this code."
    #     stat.value = _FAILED
    #     return 
    
    if not dtype or dtype != "unittest" : 
        dtype = "none"

    with safe_environment(), create_tempdir(dtype):
        
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil
        import builtins
        
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        # Disable functionalities that can make destructive changes to the test.
        reliability_guard(max_as_limit, max_data_limit, max_stack_limit)
        
        module_name = "__test__"
        new_module = types.ModuleType(module_name)
        # Set necessary attributes for the module
        new_module.__dict__.update({
            '__builtins__': builtins,
            '__file__': f"{module_name}.py",
            '__package__': None,
            '__doc__': None,
            'sys': sys,
            'os': os,
            'environ': os.environ,
        })

        try:
            full_code = code + "\n" + test_code

            with swallow_io():
                exec(compile(full_code, f"{module_name}.py", 'exec'), new_module.__dict__)
                sys.modules[module_name] = new_module
                TestCases = getattr(new_module, 'TestCases')
                loader = unittest.TestLoader()
                suite = loader.loadTestsFromTestCase(TestCases)
                test_result = unittest.TestResult()
                start_time = time.time()
                with time_limit(timeout):
                    suite.run(test_result)
            
            issues = test_result.failures + test_result.errors
            for test, trace in issues:
                details[test.id().split(".")[-1]] = trace
            stat.value = _SUCCESS
        except BaseException as e:
            details["ALL"] = str(e)
            stat.value = _FAILED
            
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        
def untrusted_check(
    code: str,
    test_code: str,
    entry_point: str,
    max_as_limit: float,
    max_data_limit: float,
    max_stack_limit: float, 
    min_time_limit: float = 10,
    gt_time_limit: float = 60, 
    dtype: str = "None"
) -> Tuple[str, np.ndarray]:
    
    time_limit = max(min_time_limit, gt_time_limit)
    timeout = max(os.getenv("BIGCODEBENCH_TIMEOUT_PER_TASK", TIMEOUT_LIMIT), time_limit) + 1
    # shared memory objects
    stat = Value("i", _UNKNOWN)
    manager = Manager()
    details = manager.dict()

    p = multiprocessing.Process(
        target=unsafe_execute,
        args=(
            entry_point,
            code,
            test_code,
            timeout,
            max_as_limit,
            max_data_limit,
            max_stack_limit,
            stat,
            details,
            dtype
        ),
    )
    p.start()
    p.join(timeout=timeout+1)
    if p.is_alive():
        p.terminate()
        time.sleep(0.1)
    if p.is_alive():
        p.kill()
        time.sleep(0.1)

    stat = _mapping[stat.value]
    # convert details to a dict
    details = dict(details)
    
    if not stat:
        stat = TIMEOUT
    if stat == PASS:
        if details:
            stat = FAIL

    return stat, details

