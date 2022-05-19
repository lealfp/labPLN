import torch, csv, logging, math
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, models, InputExample, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
logging.info(str(n_gpu)+ ' '+ str(device) + ' Device available: '+ torch.cuda.get_device_name(0))

torch.cuda.empty_cache()

MAX_SEQ_LENGTH = 128
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 2

DATASET = '../../datasets/STS/stsbenchmark.tsv'
MODEL_PATH = 'dmis-lab/biobert-base-cased-v1.1'
# OUTPUT_MODEL = 'output/' + MODEL_PATH + '/' + str(NUM_EPOCHS) + 'epochs'

word_embedding_model = models.Transformer(MODEL_PATH, max_seq_length=MAX_SEQ_LENGTH)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
# model.save(OUTPUT_MODEL)

train_samples = []
dev_samples = []

with open(DATASET, "r") as f:
    reader = csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0

        if row['split'] == 'train':
            train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
        if row['split'] == 'dev':
            dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=TRAIN_BATCH_SIZE)
train_loss = losses.MultipleNegativesRankingLoss(model)
warmup_steps = math.ceil(len(train_dataloader) * NUM_EPOCHS * 0.05)

dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=NUM_EPOCHS,
          evaluation_steps=len(train_samples),
          warmup_steps=warmup_steps,
          # output_path=OUTPUT_MODEL,
          # save_best_model = True,
          # checkpoint_path=OUTPUT_MODEL,
          # checkpoint_save_steps=10*len(train_dataloader),
)
