<p align="center">
  <h1 align="center">UnitCoder: Scalable Code Synthesis from Pre-training Corpora</h1>
  <p align="center">
    <a href="https://arxiv.org/pdf/2502.11460">
      <img src='https://img.shields.io/badge/Paper-PDF-red?style=flat&logo=arXiv&logoColor=red' alt='Paper PDF'>
    </a>
    <a href='https://github.com/Entarochuan/UnitCoder' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Code-GitHub-green?style=flat&logo=github&logoColor=green' alt='Code GitHub'>
    </a>
    <!-- <a href='https://huggingface.co/unitcoder' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Model-Hugging%20Face-yellow?style=flat&logo=Hugging%20face&logoColor=yellow' alt='Model Hugging Face'>
    </a> -->
  </p>
</p>

UnitCoder is a novel framework for scalable iterative code synthesis that leverages unit test guidance to generate high-quality, executable code from pre-training corpora. Our approach combines code filtering, unit test execution sandbox, and iterative refinement to produce high-quality post training code dataset.

<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#news">News</a>
    </li>
    <li>
      <a href="#requirements">Requirements</a>
    </li>
    <li>
      <a href="#framework">Framework Overview</a>
    </li>
    <li>
      <a href="#usage">Usage</a>
      <ul>
        <li>
          <a href="#code-filtering">Code Filtering</a>
        </li>
        <li>
          <a href="#code-synthesis">Code Synthesis</a>
        </li>
        <li>
          <a href="#agent-configuration">Agent Configuration</a>
        </li>
        <li>
          <a href="#code-polishing">Code Polishing</a>
        </li>
      </ul>
    </li>
    <li>
      <a href="#demo">Demo</a>
    </li>
    <li>
      <a href="#data">Data</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
    <li>
      <a href="#acknowledgements">Acknowledgements</a>
    </li>
  </ol>
</details>

## 1. News

- [2025-08-27] We have released the whole pipeline of UnitCoder.
- [2025-08-21] UnitCoder paper is accepted to EMNLP 2025.

## 2. Requirements

### 2.1 Install Sandbox Execution dependencies

```bash
pip install -r requirements-eval.txt
```

### 2.2 Install inference dependencies

```bash
pip install openai aiofiles
```


## 3. Framework Overview

UnitCoder consists of four main components:

1. **Code Filtering**: Filters executable functions from pre-training corpora.
2. **Unit Test Execution Sandbox**: Sandbox for safe unit test execution.
3. **Iterative Refinement Module**: Multi-agent system for test generation and debugging.
4. **Code Polishing Module**: Post-processing for enhanced code quality.

## 4. Usage

### 4.1 Code Filtering

Prepare your data by filtering code from pre-training corpora:

```bash
bash filter_code.sh
```

**Input/Output Format**:
- **Input**: JSONL files with `id` and `content` fields, see `data/raw/demo_data.jsonl`.
- **Output**: JSONL files with `id`, `content`, and `import_code` fields
  - `content`: AST-processed functions
  - `import_code`: Required import statements

### 4.2 Code Synthesis

Run the iterative code synthesis pipeline:

```bash
bash run.sh
```

### 4.3 Agent Configuration {#agent-configuration}

**Agent Configuration**:
- **Test Generation Agent**: Generates unit tests.
- **Debug Agent**: Iteratively refines code based on test execution results.

### 4.4 Code Polishing

Post-process generated code for:
- Comment addition
- Function header generation
- Code style standardization

## 6. Demo

Check the `data/demo` directory for examples of:
- Filtered code samples
- Synthesis pipeline outputs
- Final polished results

## 7. Citation

If you find our work useful, please consider citing:

```bibtex
@article{ma2025unitcoder,
  title={UnitCoder: Scalable Iterative Code Synthesis with Unit Test Guidance},
  author={Ma, Yichuan and Shao, Yunfan and Li, Peiji and Song, Demin and Guo, Qipeng and Li, Linyang and Qiu, Xipeng and Chen, Kai},
  journal={arXiv preprint arXiv:2502.11460},
  year={2025}
}
```

## 8. Acknowledgements

We are grateful to the open-source community for their contributions to code generation and evaluation research. We would like to thank the following works for their code and methods:

- **[Case2Code](https://github.com/choosewhatulike/case2code)**: For providing the foundation of code synthesis framework and evaluation methodology
- **[BigCodeBench](https://github.com/bigcode-project/bigcodebench)**: For the comprehensive code evaluation benchmark and unit test execution infrastructure

