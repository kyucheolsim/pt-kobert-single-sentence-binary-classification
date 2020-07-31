###
### KoBERT(https://github.com/SKTBrain/KoBERT)와 Naver Sentiment Analysis Fine-Tuning with pytorch(https://colab.research.google.com/github/SKTBrain/KoBERT/blob/master/scripts/NSMC/naver_review_classifications_pytorch_kobert.ipynb)를 이용하여 간단한 Text Classification 예제 구현
###

import torch
import numpy as np
import gluonnlp as nlp
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from KoBERTClassifier import BERTClassifier

model_path = "./model-nsmc.pt"
sample_path = "./sample.txt"
device = "cpu"

bertmodel, vocab = get_pytorch_kobert_model()
model = BERTClassifier(bertmodel).to(device)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint)
model.eval()

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
transform = nlp.data.BERTSentenceTransform(tok, max_seq_length=64, pad=True, pair=False)
for sample in open(sample_path).readlines():
	sample = sample.strip()
	transformed = transform([sample])
	(token_ids, valid_len, type_ids) = transformed
	token_ids = torch.LongTensor(token_ids).unsqueeze(0).to(device)
	type_ids = torch.LongTensor(type_ids).unsqueeze(0).to(device)
	valid_len = torch.tensor(valid_len, dtype=torch.int32).unsqueeze(0).to(device)
	output = model(token_ids, valid_len, type_ids)
	#print(output.softmax(-1))
	_, pred = torch.max(output, 1)
	print("{} => {}".format(sample, pred.item()))
