import csv, json, math, logging, torch
from torch.utils.data import DataLoader
from sentence_transformers import InputExample, SentenceTransformer, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

NUM_EPOCHS = 3
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 128

DATASET_PATH = "../../datasets/NLI/mNLI/"
MODEL_PATH = "dmis-lab/biobert-base-cased-v1.1"
OUTPUT_MODEL = "output/" + MODEL_PATH

model = models.Transformer(MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH)
pooling_model = models.Pooling(model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[model, pooling_model])
model.save(OUTPUT_MODEL)

train_samples, dev_samples, test_samples = [],[],[]
with open(DATASET_PATH + 'mli_train_v1.jsonl', "r") as f:
    json_list = list(f)
    for j in json_list:
        row = json.loads(j)
        gold_label = row['gold_label']
        float_gold_label = float(0)

        if gold_label == "contradiction": float_gold_label = 0 / 3.0
        if gold_label == "entailment": float_gold_label = 1 / 3.0
        if gold_label == "neutral": float_gold_label = 2 / 3.0

        inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=float_gold_label)
        train_samples.append(inp_example)

with open(DATASET_PATH + "mli_dev_v1.jsonl", "r") as f:
    json_list = list(f)
    for j in json_list:
        row = json.loads(j)
        gold_label = row["gold_label"]
        float_gold_label = float(0)

        if gold_label == "contradiction": float_gold_label = 0 / 3.0
        if gold_label == "entailment":  float_gold_label = 1 / 3.0
        if gold_label == "neutral": float_gold_label = 2 / 3.0

        inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=float_gold_label)
        dev_samples.append(inp_example)

with open(DATASET_PATH + "mli_test_v1.jsonl", "r") as f:
    json_list = list(f)
    for j in json_list:
        row = json.loads(j)

        gold_label = row["gold_label"]
        float_gold_label = float(0)

        if gold_label == "contradiction": float_gold_label = 0 / 3.0
        if gold_label == "entailment": float_gold_label = 1 / 3.0
        if gold_label == "neutral": float_gold_label = 2 / 3.0

        inp_example = InputExample(texts=[row["sentence1"], row["sentence2"]], label=float_gold_label)
        test_samples.append(inp_example)

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.CosineSimilarityLoss(model=model)
warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.1)
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="dev")

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=NUM_EPOCHS,
          evaluation_steps=len(train_samples),
          warmup_steps=warmup_steps,
          output_path=OUTPUT_MODEL,
          save_best_model = True)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=BATCH_SIZE, name="test")
test_evaluator(model, output_path=OUTPUT_MODEL)
