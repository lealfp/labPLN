import torch, csv, logging, math
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

DATASET = "../../datasets/STS/stsbenchmark.tsv"
MODEL_PATH = "bert-base-cased"
OUTPUT_MODEL = "output/" + MODEL_PATH

NUM_EPOCHS = 1
BATCH_SIZE = 2
MAX_SEQ_LENGTH = 128

word_embedding_model = models.Transformer(MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.save(OUTPUT_MODEL)

train_samples, dev_samples, test_samples = [],[],[]
with open(DATASET, 'r') as f:
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row["score"]) / 5.0
        if row["split"] == "train": train_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))
        if row["split"] == "dev": dev_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))
        if row["split"] == "test": test_samples.append(InputExample(texts=[row["sentence1"], row["sentence2"]], label=score))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.05)

evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name="dev")

model.fit(train_objectives=[(train_dataloader, train_loss)],
    evaluator=evaluator,
    epochs=NUM_EPOCHS,
    evaluation_steps=len(train_samples),
    warmup_steps=warmup_steps,
    output_path=OUTPUT_MODEL,
    save_best_model = True,
)

test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=BATCH_SIZE, name="test")
test_evaluator(model, output_path=OUTPUT_MODEL)
