###
### KoBERT(https://github.com/SKTBrain/KoBERT)와 Naver Sentiment Analysis Fine-Tuning with pytorch(https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)를 이용하여 간단한 Text Classification 예제 구현
### Naver sentiment movie corpus v1.0 (https://github.com/e9t/nsmc)
###

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
try:
	from transformers.optimization import WarmupLinearSchedule
except:
	from transformers.optimization import get_linear_schedule_with_warmup as WarmupLinearSchedule

from KoBERTClassifier import calc_accuracy, BERTDataset, BERTClassifier

DEBUG = True

## Setting parameters
max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 15
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5
model_path = "./model-nsmc.pt"


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)
print("device count: ", torch.cuda.device_count())

device = torch.device("cuda")
print("device: ", device)

# download: pytorch_kobert(BertModel), tokenizer(vocab_file: pretrained subword tokenization model)
# BertModel(transformers), nlp.vocab.BERTVocab.from_sentencepiece(vocab_file...)
bertmodel, vocab = get_pytorch_kobert_model()
#print(bertmodel)
print(vocab)
print("vocab size: ", len(vocab))
print("10 tokens: ", vocab.idx_to_token[:10])
print("reserved tokens", vocab.reserved_tokens)

#print("word embedding size: ", bertmodel.embeddings.word_embeddings.weight.size())
#print("[0,:] embedding: ", bertmodel.embeddings.word_embeddings.weight[0, :5])
#pad_idx = torch.LongTensor([0])
#print("[0,:] embedding: ", bertmodel.embeddings.word_embeddings(pad_idx)[0, :5])

# downloaded tokenizer file path(vocab_file)
tokenizer = get_tokenizer()
#print(tokenizer)

# path to the pretrained subword tokenization model, BERTVocab
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

#sample_sent = "Hello World"
#sample_tokens = tok(sample_sent)
#print(sample_tokens)
#print(tok.convert_tokens_to_ids(sample_tokens))

if DEBUG:
	dataset_train = [['좋아요 good', '1'], ['별로에요', '0'], ['좋아 nice', '1'], ['별로', '0']]
	dataset_test = [['좋아요', '1'], ['별로군', '0'], ['좋은듯', '1'], ['별로에요', '0']]
	#print(tok(dataset_test[0][0]))
else:
	# id document label
	dataset_train = nlp.data.TSVDataset("./data/ratings_train.txt", field_indices=[1,2], num_discard_samples=1)
	dataset_test = nlp.data.TSVDataset("./data/ratings_test.txt", field_indices=[1,2], num_discard_samples=1)

'''
## pair=True
#dataset_train_pair = [[['좋아요', 'good'], '1'], [['별로에요', 'good'], '0']]
#data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, True)
#data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, True)
'''

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=4)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=4)

train_size = len(train_dataloader.dataset)
test_size = len(test_dataloader.dataset)
print("train size: {}, test size: {}".format(train_size, test_size))

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
#for n, p in model.named_parameters():
#	print(n)

# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
	{'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
	{'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()
t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)
print("total steps: {}, warmup steps: {}".format(t_total, warmup_step))

try:
	scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_step, t_total=t_total)
except:
	scheduler = WarmupLinearSchedule(optimizer, warmup_step, t_total)

best_acc = 0.0
for e in range(num_epochs):
	train_acc = 0.0
	test_acc = 0.0
	model.train()
	for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
		optimizer.zero_grad()
		token_ids = token_ids.long().to(device)
		segment_ids = segment_ids.long().to(device)
		valid_length = valid_length.to(device)
		label = label.long().to(device)
		out = model(token_ids, valid_length, segment_ids)
		loss = loss_fn(out, label)
		loss.backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
		optimizer.step()
		scheduler.step()  # Update learning rate schedule
		train_acc += calc_accuracy(out, label)
		if batch_id % log_interval == 0:
			print("epoch {}, batch id {}, loss {:.9f}, train acc {:.9f}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
	print("- epoch {}, train acc {:.9f}".format(e+1, train_acc / (batch_id+1)))

	model.eval()
	for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
		token_ids = token_ids.long().to(device)
		segment_ids = segment_ids.long().to(device)
		valid_length= valid_length.to(device)
		label = label.long().to(device)
		out = model(token_ids, valid_length, segment_ids)
		test_acc += calc_accuracy(out, label)
	test_acc = test_acc / (batch_id+1)
	print("- epoch {}, test acc {:.9f}".format(e+1, test_acc))

	if test_acc > best_acc:
		best_acc = test_acc
		torch.save(model.state_dict(), model_path)
		print("* epoch {}, best acc {:.9f}".format(e+1, best_acc))

