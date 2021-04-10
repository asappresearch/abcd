import json
import random
import torch

from tqdm import tqdm as progress_bar
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

print("Loading data ...")
utt_texts = json.load(open(f'data/utterances.json', 'r'))
num_cands = len(utt_texts)
utt_vectors = []

cand_embeds, cand_segments, cand_masks = [], [], []
for cand_text in progress_bar(utt_texts, total=num_cands):
  cand_inputs = tokenizer(cand_text, return_tensors="pt")
  with torch.no_grad():
	  cand_outputs = model(**cand_inputs)
  utt_vectors.append(cand_outputs.pooler_output)

utt_vectors = torch.cat(utt_vectors)
print("utt_vectors: {}".format(utt_vectors.shape))
torch.save(utt_vectors, 'data/utt_vectors.pt')

