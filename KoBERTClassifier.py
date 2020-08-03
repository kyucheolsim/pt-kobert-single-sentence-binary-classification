###
### KoBERT(https://github.com/SKTBrain/KoBERT)와 Naver Sentiment Analysis Fine-Tuning with pytorch(https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)를 이용하여 간단한 Text Classification 예제 구현
###

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

def calc_accuracy_sum(X, Y):
	max_vals, max_indices = torch.max(X, 1)
	train_acc = (max_indices == Y).sum().data.cpu().numpy()
	return train_acc


def calc_accuracy(X, Y):
	max_vals, max_indices = torch.max(X, 1)
	train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
	return train_acc


class BERTDataset(Dataset):
	def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
				 pad, pair):
		transform = nlp.data.BERTSentenceTransform(
			bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

		#pair=True
		#dataset_train_pair = [[['좋아요', 'good'], '1']],
		#self.sentences = [transform(i[sent_idx]) for i in dataset]
		#self.sentences = [transform([i[sent_i1], i[sent_i2]]) for i in dataset]

		# (batch_size, (token_ids:max_seq_len, valid_len, type_ids:max_seq_len))
		self.sentences = [transform([i[sent_idx]]) for i in dataset]
		self.labels = [np.int32(i[label_idx]) for i in dataset]

	def __getitem__(self, i):
		return (self.sentences[i] + (self.labels[i], ))

	def __len__(self):
		return (len(self.labels))


class BERTClassifier(nn.Module):
	def __init__(self,
				 bert,
				 hidden_size = 768,
				 num_classes=2,
				 dr_rate=None,
				 params=None):
		super(BERTClassifier, self).__init__()
		self.bert = bert
		self.dr_rate = dr_rate

		self.classifier = nn.Linear(hidden_size , num_classes)
		if dr_rate:
			self.dropout = nn.Dropout(p=dr_rate)

	def gen_attention_mask(self, token_ids, valid_length):
		attention_mask = torch.zeros_like(token_ids)
		for i, v in enumerate(valid_length):
			if v.ndimension() == 0:
				v = v.unsqueeze(0)
			attention_mask[i][:v] = 1
		return attention_mask.float()

	def forward(self, token_ids, valid_length, segment_ids):
		attention_mask = self.gen_attention_mask(token_ids, valid_length)

		_, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
		if self.dr_rate:
			out = self.dropout(pooler)
		else:
			out = pooler
		return self.classifier(out)

