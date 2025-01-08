Execute on terminal:

```export TASK=natural-language-inference

cd /app/src/${TASK}
pip install torch
pip install sentence_transformers

python finetune.py
```
