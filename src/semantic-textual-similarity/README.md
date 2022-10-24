export TASK=semantic-textual-similarity

cd /app/src/${TASK}

pip install torch
pip install sentence_transformers

python finetune.py
