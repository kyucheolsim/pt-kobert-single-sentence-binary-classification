###
### KoBERT(https://github.com/SKTBrain/KoBERT)와 Naver Sentiment Analysis Fine-Tuning with pytorch(https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)를 이용하여 간단한 Text Classification 예제 구현
### Naver sentiment movie corpus v1.0 (https://github.com/e9t/nsmc)
###

import torch
#from torch import nn
#import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from KoBERTClassifier import calc_accuracy_sum, BERTDataset, BERTClassifier

# 25000 items, cpu: 09:47 (587), gpu: 32 (x18.34)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device: ", device)
#print("device count: ", torch.cuda.device_count())

## Setting parameters
max_len = 64
batch_size = 64

# download: pytorch_kobert(BertModel), tokenizer(vocab_file: pretrained subword tokenization model)
# BertModel(transformers), nlp.vocab.BERTVocab.from_sentencepiece(vocab_file...)
bertmodel, vocab = get_pytorch_kobert_model()
#print(bertmodel)
#print(vocab)
#print("vocab size: ", len(vocab))
#print("10 tokens: ", vocab.idx_to_token[:10])
#print("reserved tokens", vocab.reserved_tokens)

model = BERTClassifier(bertmodel).to(device)
#for n, p in model.named_parameters():
#	print(n)
model_path = "./model-nsmc-e5-a0.89.pt"
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dic'])
model.eval()

# downloaded tokenizer file path(vocab_file)
tokenizer = get_tokenizer()
#print(tokenizer)

# path to the pretrained subword tokenization model, BERTVocab
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

#sample_sent = "Hello World"
#sample_tokens = tok(sample_sent)
#print(sample_tokens)
#print(tok.convert_tokens_to_ids(sample_tokens))

'''
## pair=True
#dataset_train_pair = [[['좋아요', 'good'], '1'], [['별로에요', 'good'], '0']]
#data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, True)
'''

# id document label
dataset_test = nlp.data.TSVDataset("./data/ratings_hidden_test.txt", field_indices=[1,2], num_discard_samples=1)
print("test set: ", np.shape(dataset_test))

data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=4)

test_size = len(test_dataloader.dataset)
print("test size: {}".format(test_size))
test_acc = 0.0
for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
	token_ids = token_ids.long().to(device)
	segment_ids = segment_ids.long().to(device)
	valid_length= valid_length.to(device)
	label = label.long().to(device)
	out = model(token_ids, valid_length, segment_ids)
	test_acc += calc_accuracy_sum(out, label)
test_acc = test_acc / (test_size)
print("- test acc {:.9f}".format(test_acc))
