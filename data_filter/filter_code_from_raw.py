import os
import json
from tqdm import tqdm
from ast_utils import *
import argparse
from multiprocessing import Pool, Process, Queue
import re
import copy
import uuid

ALLOWED_MODULES = set(['random', 'itertools', 'collections', 'string', 'statistics', 'numpy', 'math', 're', 'os', 'operator', 'csv', 'pandas', 'matplotlib', 'seaborn', 'subprocess', 'datetime', 'json', 'ftplib', 'configparser', 'shutil', 'glob', 'psutil', 'time', 'zipfile', 'ast', 'platform', 'base64', 'hashlib', 'zlib', 'cryptography', 'requests', 'sklearn', 'nltk', 'bs4', 'functools', 'wordcloud', 'scipy', 'pytz', 'regex', 'wikipedia', 'sqlite3', 'socket', 'django', 'binascii', 'io', 'logging', 'flask', 'flask_restful', 'flask_login', 'flask_wtf', 'wtforms', 'werkzeug', 'flask_mail', 'statsmodels', 'ipaddress', 'threading', 'mimetypes', 'urllib', 'gzip', 'struct', 'holidays', 'PIL', 'skimage', 'uuid', 'folium', 'geopy', 'shapely', 'geopandas', 'smtplib', 'heapq', 'bisect', 'multiprocessing', 'cv2', 'turtle', 'librosa', 'soundfile', 'http', 'cgi', 'email', 'sympy', 'mpl_toolkits', 'mechanize', 'tensorflow', 'dateutil', 'ssl', 'python_http_client', 'sendgrid', 'sys', 'pathlib', 'queue', 'hmac', 'blake3', 'signal', 'cmath', 'docx', 'openpyxl', 'texttable', 'textblob', 'natsort', 'unicodedata', 'codecs', 'keras', 'pickle', 'decimal', 'enum', 'xmltodict', 'faker', 'calendar', 'xlwt', 'rsa', 'difflib', 'secrets', 'importlib', 'pkgutil', 'prettytable', 'ctypes', 'types', 'inspect', 'Crypto', 'pyquery', 'array', 'typing', 'gensim', 'fnmatch', 'yaml', 'html', 'textwrap', 'tarfile', 'errno', 'Levenshtein', 'warnings', 'wordninja', 'lxml', 'xml', 'pytesseract', 'chardet', 'select', 'getpass', 'shlex']
)

def check_code_validation(ex):
    ex_id = ex['id']
    tree = source_to_ast(ex['content']) 
    if tree is None:
        return False, None
    results = []
    for idx, func in enumerate(get_functions(tree)):
        func_tree = FunctionAst(func)
        if not func_tree.has_return():
            continue
        if func_tree.has_import():
            continue
        if func_tree.is_nested_function():
            continue
        if not func_tree.has_args():
            continue
        ex['id'] = f'{ex_id}_{idx}'
        ex['content'] = ast_to_source(func_tree.node)
        results.append(copy.deepcopy(ex))
    if len(results) == 0:
        return False, None
    return True, results


def check_code_validation_v2(ex):
    content = ex.get('content', None)
    if content is None:
        return False, None
    
    ex_id = ex.get('code_id', None)
    if ex_id is None:
        ex_id = uuid.uuid3(uuid.NAMESPACE_DNS, content)
    
    save_ex = dict(
        id = None, 
        content = "",
        import_code = []
    )
    
    tree = source_to_ast(ex['content'])
    if tree is None:
        return False, None
    
    results = []
    
    # Get all imports from the entire code
    all_imports, original_imports = get_imports(tree)
    
    for idx, func in enumerate(get_functions(tree)):
        func_tree = FunctionAst(func)
        if not func_tree.has_return() or func_tree.has_import() or func_tree.is_nested_function() or not func_tree.has_args():
            continue
        
        # Get used names in the function
        used_names = get_used_names(func_tree.node)
        
        # Check if imports are used and allowed
        func_imports = set()
        func_original_imports = []
        for used_name in used_names:
            if used_name in all_imports:
                full_import = all_imports[used_name]
                root_module = full_import.split('.')[0]
                if root_module not in ALLOWED_MODULES:
                    return False, None
                func_imports.add(root_module)
                # Find and add the original import statement
                for orig_import in original_imports:
                    if used_name in orig_import or full_import in orig_import:
                        func_original_imports.append(orig_import)
        
        save_ex['id'] = f'{ex_id}_{idx}'
        save_ex['content'] = ast_to_source(func_tree.node)
        if save_ex['content'] == '':
            continue
        
        save_ex['import_code'] = func_original_imports
        # save_ex['original_imports'] = func_original_imports
        results.append(copy.deepcopy(save_ex))
    
    if len(results) == 0:
        return False, None
    
    return True, results


def process_one_file(in_file, out_file, n_workers=4):
    with open(in_file, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    results = []
    pool = Pool(n_workers)
    results = []
    for ex in data:
        # r = pool.apply_async(check_code_validation, (ex,))
        r = pool.apply_async(check_code_validation_v2, (ex,))
        results.append((ex, r))
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for ex, r in tqdm(results):
            check_status, new_ex_list = r.get()
            if check_status:
                for new_ex in new_ex_list:
                    f.write(json.dumps(new_ex) + '\n')
    pool.close()

def main_old():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_workers', type=int, default=4)

    args = parser.parse_args()
    if os.path.isdir(args.input):
        data = []
        for fn in os.listdir(args.input):
            with open(os.path.join(args.input, fn), 'r', encoding='utf-8') as f:
                data += [json.loads(l) for l in f]
    else:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = [json.loads(l) for l in f]
        
    pool = Pool(args.n_workers)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    results = []
    for ex in data:
        # r = pool.apply_async(check_code_validation, (ex,))
        r = pool.apply_async(check_code_validation_v2, (ex,))
        results.append((ex, r))
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for ex, r in tqdm(results):
            check_status, new_ex_list = r.get()
            if check_status:
                for new_ex in new_ex_list:
                    f.write(json.dumps(new_ex) + '\n')
                    
                    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--n_workers', type=int, default=4)

    args = parser.parse_args()
    if os.path.isdir(args.input):
        in_files = [os.path.join(args.input, fn) for fn in os.listdir(args.input) if fn.endswith('.jsonl')]
        in_files = sorted(in_files)
        out_files = [os.path.join(args.output, os.path.basename(fn)) for fn in in_files]
    else:
        in_files = [args.input]
        out_files = [args.output]
        
    for in_file, out_file in zip(in_files, out_files):
        print(f'Processing {in_file} -> {out_file}')
        process_one_file(in_file, out_file, args.n_workers)

if __name__ == '__main__':
    main()
#     code = """
# from coco import bunny
# from coco.bungy import carrot
# from meta import teta
# from rocket import spaceship as sp
# import bingo
# import com.stackoverflow
# import motorbike as car
# import module1, module2

# s="a random variable"

# def func():
#     a = 1 + 2
#     return a
# """

#     code = """
# import random
# import numpy as np
# from matplotlib import pyplot as plt
# import glob
# import string
# def func(args):
#     A = random.random()
#     a = 1 + 2
#     plt.show()
#     B = np.zeros(5)
#     return a
# """

#     # node = source_to_ast(code)
#     # for i in get_imports(node):
#     #     print('>>>', i[0], 'm=', i[1])
        
#     print(check_code_validation_v2(dict(
#         ex_id = 1,
#         content=code
#     )))
