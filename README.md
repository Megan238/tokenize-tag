# Tokenize & Tag API

## Overview
A **multilingual keyword tokenization and tagging service** for e-commerce search terms.

Features:
- Dictionary + trie-based tokenization
- LLM fallback for low-confidence tokens
- Concurrent-safe FastAPI service
- Supports Japanese, English, Spanish, and French

---

## Requirements & Setup

### Environment
```bash
python >= 3.9
```

### Install Dependencies
```bash
pip install fastapi uvicorn openai
```

### Set OpenAI API Key
The system uses OpenAI for LLM fallback classification.

```bash
export OPENAI_API_KEY="your_api_key_here"
```

(Windows PowerShell)
```powershell
setx OPENAI_API_KEY "your_api_key_here"
```
### Data
The `keywords.csv` file was provided as part of the take-home assignment
and is excluded from this public repository to respect data ownership.

To run locally, place `keywords.csv` in the project root.

---

## Run the API

```bash
python -m uvicorn app:app --reload
```

API documentation:
```
http://127.0.0.1:8000/docs
```

---

## Run Tests / Demo

### Batch Test
```bash
python test.py
```

- Input: `keywords.csv`
- Output: `keywords_test_output.json` (UTF-8 encoded test results)

### Local Demo
You can also run the demo directly using the `main` function in `tag.py`.

---

