from transformers import RobertaTokenizer, RobertaModel 
from copy import deepcopy
import openai
import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
import custom_datasets
from multiprocessing.pool import ThreadPool
import time
from MMD_calculate import mmd
from sklearn import metrics
from meta_train import mmdPreModel, Meta, Bert_Transformer_Layer
from utils_MMD import MMDu, MMD_batch2, TST_MMD_u
from MGTBenchold import dataset_loader
from collections import namedtuple
from dataTST import get_metadata
import sys
import math
import nltk
# if not os.path.exists(nltk.data.find('tokenizers/punkt')):
# 	nltk.download('punkt')
# else:
# 	print("punkt has already been downloaded")

# define regex to match all <extra_id_*> tokens, where * is an integer
pattern = re.compile(r"<extra_id_\d+>")

model_path_dit={
    'gpt2': '/mnt/cephfs/dataset/zhangshuhai/backup20240107/detect-gpt-tmp/pretrain_model/gpt2',
	'roberta-base-openai-detector': '/mnt/cephfs/dataset/zhangshuhai/backup20240107/detect-gpt-tmp/pretrain_model/roberta-base-openai-detector',
	'EleutherAI/gpt-neo-125m':'/mnt/cephfs/dataset/zhangshuhai/backup20240107/detect-gpt-tmp/pretrain_model/gpt-neo-125m',
	'minhtoan/gpt3-small-finetune-cnndaily-news':'/mnt/cephfs/dataset/zhangshuhai/backup20240107/detect-gpt-tmp/pretrain_model/gpt3-small-finetune-cnndaily-news',
	't5-large':'/mnt/cephfs/dataset/zhangshuhai/backup20240107/detect-gpt-tmp/pretrain_model/t5-large',
	't5-small':'/mnt/cephfs/dataset/zhangshuhai/backup20240107/detect-gpt-tmp/pretrain_model/t5-small',
}

def plot_mi(clean, adv):

	mi_nat = clean.numpy()
	label_clean = 'Clean'

	mi_svhn = adv.numpy()
	label_adv = 'Adv'

	mi_nat = mi_nat[~np.isnan(mi_nat)]
	mi_svhn = mi_svhn[~np.isnan(mi_svhn)]

	x = np.concatenate((mi_nat, mi_svhn), 0)
	y = np.zeros(x.shape[0])
	y[mi_nat.shape[0]:] = 1

	ap = metrics.roc_auc_score(y, x)
	fpr, tpr, thresholds = metrics.roc_curve(y, x)
	accs = {th: tpr[np.argwhere(fpr <= th).max()] for th in [0.01, 0.05, 0.1]}
	print("auroc: {:.4f}; ".format(ap) + "; ".join(["TPR: {:.4f} @ FPR={:.4f}".format(v, k) for k, v in accs.items()]) + "  {}-{}".format(len(mi_nat), len(mi_svhn)))
	return ap

def load_base_model():
	print('MOVING BASE MODEL TO GPU...', end='', flush=True)
	start = time.time()
	try:
		mask_model.cpu()
	except NameError:
		pass
	if args.openai_model is None:
		base_model.to(DEVICE)
	print(f'DONE ({time.time() - start:.2f}s)')


def load_mask_model():
	print('MOVING MASK MODEL TO GPU...', end='', flush=True)
	start = time.time()

	if args.openai_model is None:
		base_model.cpu()
	if not args.random_fills:
		mask_model.to(DEVICE)
	print(f'DONE ({time.time() - start:.2f}s)')


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
	tokens = text.split(' ')
	mask_string = '<<<mask>>>'

	n_spans = pct * len(tokens) / (span_length + args.buffer_size * 2)
	if ceil_pct:
		n_spans = np.ceil(n_spans)
	n_spans = int(n_spans)

	n_masks = 0
	while n_masks < n_spans:
		start = np.random.randint(0, len(tokens) - span_length)
		end = start + span_length
		search_start = max(0, start - args.buffer_size)
		search_end = min(len(tokens), end + args.buffer_size)
		if mask_string not in tokens[search_start:search_end]:
			tokens[start:end] = [mask_string]
			n_masks += 1
	
	# replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
	num_filled = 0
	for idx, token in enumerate(tokens):
		if token == mask_string:
			tokens[idx] = f'<extra_id_{num_filled}>'
			num_filled += 1
	assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
	text = ' '.join(tokens)
	return text


def count_masks(texts):
	return [len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts]


# replace each masked span with a sample from T5 mask_model
def replace_masks(texts):
	n_expected = count_masks(texts)
	stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
	tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
	outputs = mask_model.generate(**tokens, max_length=150, do_sample=True, top_p=args.mask_top_p, num_return_sequences=1, eos_token_id=stop_id)
	return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
	# remove <pad> from beginning of each text
	texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

	# return the text in between each matched mask token
	extracted_fills = [pattern.split(x)[1:-1] for x in texts]

	# remove whitespace around each fill
	extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

	return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
	# split masked text into tokens, only splitting on spaces (not newlines)
	tokens = [x.split(' ') for x in masked_texts]

	n_expected = count_masks(masked_texts)

	# replace each mask token with the corresponding fill
	for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
		if len(fills) < n:
			tokens[idx] = []
		else:
			for fill_idx in range(n):
				text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

	# join tokens back into text
	texts = [" ".join(x) for x in tokens]
	return texts


def perturb_texts_(texts, span_length, pct, ceil_pct=False):
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
    
    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while '' in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == '']
        print(f'WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}].')
        masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for idx, x in enumerate(texts) if idx in idxs]
        raw_fills = replace_masks(masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1

    return perturbed_texts

def perturb_texts(texts, span_length, pct, ceil_pct=False):
	chunk_size = args.chunk_size
	if '11b' in mask_filling_model_name:
		chunk_size //= 2

	outputs = []
	for i in tqdm.tqdm(range(0, len(texts), chunk_size), desc="Applying perturbations"):
		outputs.extend(perturb_texts_(texts[i:i + chunk_size], span_length, pct, ceil_pct=ceil_pct))
	return outputs


def drop_last_word(text):
	return ' '.join(text.split(' ')[:-1])


def _openai_sample(p):
	if args.dataset != 'pubmed':  # keep Answer: prefix for pubmed
		p = drop_last_word(p)

	# sample from the openai model
	kwargs = { "engine": args.openai_model, "max_tokens": 200 }
	if args.do_top_p:
		kwargs['top_p'] = args.top_p
	
	r = openai.Completion.create(prompt=f"{p}", **kwargs)
	return p + r['choices'][0].text


# sample from base_model using ****only**** the first 30 tokens in each example as context
def sample_from_model(texts, min_words=55, prompt_tokens=30):
	# encode each text as a list of token ids
	if args.dataset == 'pubmed':
		texts = [t[:t.index(custom_datasets.SEPARATOR)] for t in texts]
		all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
	else:
		all_encoded = base_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
		all_encoded = {key: value[:, :prompt_tokens] for key, value in all_encoded.items()}

	if args.openai_model:
		# decode the prefixes back into text
		prefixes = base_tokenizer.batch_decode(all_encoded['input_ids'], skip_special_tokens=True)
		pool = ThreadPool(args.batch_size)

		decoded = pool.map(_openai_sample, prefixes)
	else:
		decoded = ['' for _ in range(len(texts))]

		# sample from the model until we get a sample with at least min_words words for each example
		# this is an inefficient way to do this (since we regenerate for all inputs if just one is too short), but it works
		tries = 0
		while (min(len(x.split()) for x in decoded)) < min_words:
			m = min(len(x.split()) for x in decoded)
			if tries != 0:
				print()
				print(f"min words: {m}, needed {min_words}, regenerating (try {tries})")

			sampling_kwargs = {}
			if args.do_top_p:
				sampling_kwargs['top_p'] = args.top_p
			elif args.do_top_k:
				sampling_kwargs['top_k'] = args.top_k
			min_length = 50 if args.dataset in ['pubmed'] else 150
			outputs = base_model.generate(**all_encoded, min_length=min_length, max_length=200, do_sample=True, **sampling_kwargs, pad_token_id=base_tokenizer.eos_token_id, eos_token_id=base_tokenizer.eos_token_id)
			decoded = base_tokenizer.batch_decode(outputs[:, prompt_tokens:], skip_special_tokens=True) # remove the first 30 tokens
			tries += 1

	if args.openai_model:
		global API_TOKEN_COUNTER

		# count total number of tokens with GPT2_TOKENIZER
		total_tokens = sum(len(GPT2_TOKENIZER.encode(x)) for x in decoded)
		API_TOKEN_COUNTER += total_tokens

	return decoded

# Get the log likelihood of each text under the base_model
def get_ll(text):
	if args.openai_model:        
		kwargs = { "engine": args.openai_model, "temperature": 0, "max_tokens": 0, "echo": True, "logprobs": 0}
		r = openai.Completion.create(prompt=f"<|endoftext|>{text}", **kwargs)
		result = r['choices'][0]
		tokens, logprobs = result["logprobs"]["tokens"][1:], result["logprobs"]["token_logprobs"][1:]

		assert len(tokens) == len(logprobs), f"Expected {len(tokens)} logprobs, got {len(logprobs)}"

		return np.mean(logprobs)
	else:
		with torch.no_grad():
			tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
			labels = tokenized.input_ids
			return -base_model(**tokenized, labels=labels).loss.item()

def get_joint_pros(texts, num_tokens):
	joint_pros = [get_joint_pro(text) for text in texts]
	return [x[:num_tokens] for x in joint_pros if len(x)>num_tokens]

def get_joint_pro(text):
	
	with torch.no_grad():
		tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
		logits = base_model(**tokenized).logits
		labels = tokenized.input_ids

		logits = logits.view(-1, logits.shape[-1])[:-1]
		labels = labels.view(-1)[1:]
		log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
		log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
	return log_likelihood

# get the average rank of each observed token sorted by model likelihood
def get_rank(text, log=False):
	assert args.openai_model is None, "get_rank not implemented for OpenAI models"

	with torch.no_grad():
		tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
		logits = base_model(**tokenized).logits[:,:-1]
		labels = tokenized.input_ids[:,1:]

		# get rank of each label token in the model's likelihood ordering
		matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()

		assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

		ranks, timesteps = matches[:,-1], matches[:,-2]

		# make sure we got exactly one match for each timestep in the sequence
		assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), "Expected one match per timestep"

		ranks = ranks.float() + 1 # convert to 1-indexed rank
		if log:
			ranks = torch.log(ranks)

		return ranks.float().mean().item()


# get average entropy of each token in the text
def get_entropy(text):
	assert args.openai_model is None, "get_entropy not implemented for OpenAI models"

	with torch.no_grad():
		tokenized = base_tokenizer(text, return_tensors="pt").to(DEVICE)
		logits = base_model(**tokenized).logits[:,:-1]
		neg_entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
		return -neg_entropy.sum(-1).mean().item()


def get_roc_metrics(real_preds, sample_preds):
	fpr, tpr, _ = roc_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
	roc_auc = auc(fpr, tpr)
	return fpr.tolist(), tpr.tolist(), float(roc_auc)


def get_precision_recall_metrics(real_preds, sample_preds):
	precision, recall, _ = precision_recall_curve([0] * len(real_preds) + [1] * len(sample_preds), real_preds + sample_preds)
	pr_auc = auc(recall, precision)
	return precision.tolist(), recall.tolist(), float(pr_auc)

def run_baseline_threshold_experiment(criterion_fn, name, n_samples=500):
	results = []
	for batch in tqdm.tqdm(range(n_samples // batch_size), desc=f"Computing {name} criterion"):
		original_text = data["original"][batch * batch_size:(batch + 1) * batch_size]
		sampled_text = data["sampled"][batch * batch_size:(batch + 1) * batch_size]

		for idx in range(len(original_text)):
			results.append({
				"original": original_text[idx],
				"original_crit": criterion_fn(original_text[idx]),
				"sampled": sampled_text[idx],
				"sampled_crit": criterion_fn(sampled_text[idx]),
			})

	# compute prediction scores for real/sampled passages
	predictions = {
		'real': [x["original_crit"] for x in results],
		'samples': [x["sampled_crit"] for x in results],
	}

	fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
	p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
	print(f"{name}_threshold ROC AUC: {roc_auc}, PR AUC: {pr_auc}")
	return {
		'name': f'{name}_threshold',
		'predictions': predictions,
		'info': {
			'n_samples': n_samples,
		},
		'raw_results': results,
		'metrics': {
			'roc_auc': roc_auc,
			'fpr': fpr,
			'tpr': tpr,
		},
		'pr_metrics': {
			'pr_auc': pr_auc,
			'precision': p,
			'recall': r,
		},
		'loss': 1 - pr_auc,
	}


# strip newlines from each example; replace one or more newlines with a single space
def strip_newlines(text):
	return ' '.join(text.split())


# trim to shorter length
def trim_to_shorter_length(texta, textb):
	# truncate to shorter of o and s
	shorter_length = min(len(texta.split(' ')), len(textb.split(' ')))
	texta = ' '.join(texta.split(' ')[:shorter_length])
	textb = ' '.join(textb.split(' ')[:shorter_length])
	return texta, textb

def truncate_to_substring(text, substring, idx_occurrence):
	# truncate everything after the idx_occurrence occurrence of substring
	assert idx_occurrence > 0, 'idx_occurrence must be > 0'
	idx = -1
	for _ in range(idx_occurrence):
		idx = text.find(substring, idx + 1)
		if idx == -1:
			return text
	return text[:idx]


def generate_samples(raw_data, batch_size, dataset = None):
	data = {
		"original": [],
		"sampled": [],
	}

	for batch in range(len(raw_data) // batch_size):
		print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
		original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
		sampled_text = sample_from_model(original_text, min_words=30 if dataset in ['pubmed'] else 50, prompt_tokens=20 if args.dataset in ['HC3'] else 30)

		for o, s in zip(original_text, sampled_text):
			if dataset == 'pubmed':
				s = truncate_to_substring(s, 'Question:', 2)
				o = o.replace(custom_datasets.SEPARATOR, ' ')

			o, s = trim_to_shorter_length(o, s)

			# add to the data
			data["original"].append(o)
			data["sampled"].append(s)
	
	if args.pre_perturb_pct > 0:
		print(f'APPLYING {args.pre_perturb_pct}, {args.pre_perturb_span_length} PRE-PERTURBATIONS')
		load_mask_model()
		data["sampled"] = perturb_texts(data["sampled"], args.pre_perturb_span_length, args.pre_perturb_pct, ceil_pct=True)
		load_base_model()

	return data

def generate_data(dataset, data):

	# get unique examples, strip whitespace, and remove newlines
	# then take just the long examples, shuffle, take the first 5,000 to tokenize to save time
	# then take just the examples that are <= 512 tokens (for the mask model)
	# then generate n_samples samples

	# remove duplicates from the data
	data = list(dict.fromkeys(data))  # deterministic, as opposed to set()

	# strip whitespace around each example
	data = [x.strip() for x in data]

	# remove newlines from each example
	data = [strip_newlines(x) for x in data]

	# try to keep only examples with > 250 words
	if dataset in ['writing', 'squad', 'xsum']:
		long_data = [x for x in data if 300>len(x.split()) > 150]
		if len(long_data) > 0:
			data = long_data
	else:
		assert False, f'Not approved dataset {dataset}'

	# keep only examples with <= 512 tokens according to mask_tokenizer
	# this step has the extra effect of removing examples with low-quality/garbage content
	tokenized_data = preproc_tokenizer(data)
	data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

	# print stats about remainining data
	print(f"Total number of samples: {len(data)}")
	print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

	return generate_samples(data[:n_samples], batch_size=batch_size, dataset=dataset)

def generate_fake_and_combine_real(dataset, data):

	long_data = [x for x in data if len(x.split()) > 150]
	if len(long_data) > 0:
		data = long_data

	# keep only examples with <= 512 tokens according to mask_tokenizer
	# this step has the extra effect of removing examples with low-quality/garbage content
	tokenized_data = preproc_tokenizer(data)
	data = [x for x, y in zip(data, tokenized_data["input_ids"]) if len(y) <= 512]

	# print stats about remainining data
	print(f"Total number of samples: {len(data)}")
	print(f"Average number of words: {np.mean([len(x.split()) for x in data])}")

	return generate_samples(data[:n_samples], batch_size=batch_size, dataset=dataset)

def load_HC3(data_o):
	train_real = [text for text, label in zip(data_o['train']['text'], data_o['train']['label']) if label == 0]
	train_generated = [text for text, label in zip(data_o['train']['text'], data_o['train']['label']) if label == 1]
	long_train_real = [x for x in train_real if  len(x.split()) > 150]
	long_train_generated = [x for x in train_generated if len(x.split()) > 150]

	# keep only examples with <= 512 tokens according to mask_tokenizer
	# this step has the extra effect of removing examples with low-quality/garbage content
	tokenized_data = preproc_tokenizer(long_train_real, truncation=True, max_length=preproc_tokenizer.model_max_length)
	long_train_real = [x for x, y in zip(long_train_real, tokenized_data["input_ids"]) if len(y) <= 512]
	tokenized_data = preproc_tokenizer(long_train_generated, truncation=True, max_length=preproc_tokenizer.model_max_length)
	long_train_generated = [x for x, y in zip(long_train_generated, tokenized_data["input_ids"]) if len(y) <= 512]

	# print stats about remainining data
	print(f"Total number of samples: {len(long_train_real)}")
	print(f"Average number of words: {np.mean([len(x.split()) for x in long_train_real])}")

	data = {
		"original": [],
		"sampled": [],
	}
	for o, s in zip(long_train_real, long_train_generated):

		o, s = trim_to_shorter_length(o, s)

		# add to the data
		data["original"].append(o)
		data["sampled"].append(s)

	return data


def load_base_model_and_tokenizer(name):
	if name not in ['roberta-base', 'roberta-base-openai-detector', 'Hello-SimpleAI/chatgpt-detector-roberta']:
		if args.openai_model is None:
			print(f'Loading BASE model {name}...') # print(f'Loading BASE model {args.base_model_name}...')		
			base_model_kwargs = {}
			if 'gpt-j' in name or 'neox' in name or 'gpt4' in name:
				base_model_kwargs.update(dict(torch_dtype=torch.float16))
			if 'gpt-j' in name:
				base_model_kwargs.update(dict(revision='float16'))
			# base_model = transformers.AutoModelForCausalLM.from_pretrained(name, **base_model_kwargs, cache_dir=cache_dir)
			base_model = transformers.AutoModelForCausalLM.from_pretrained(model_path_dit[name])
		else:
			base_model = None

		optional_tok_kwargs = {}
		if "facebook/opt-" in name:
			print("Using non-fast tokenizer for OPT")
			optional_tok_kwargs['fast'] = False
		if args.dataset in ['pubmed']:
			optional_tok_kwargs['padding_side'] = 'left'
		# base_tokenizer = transformers.AutoTokenizer.from_pretrained(name, **optional_tok_kwargs, cache_dir=cache_dir)
		base_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_dit[name])
		base_tokenizer.pad_token_id = base_tokenizer.eos_token_id
	elif name in ['roberta-base', 'roberta-base-openai-detector', 'Hello-SimpleAI/chatgpt-detector-roberta']:
		
		# base_tokenizer = RobertaTokenizer.from_pretrained(name, cache_dir=cache_dir)
		# base_model = RobertaModel.from_pretrained(name, output_hidden_states=True, cache_dir=cache_dir)
		base_tokenizer = RobertaTokenizer.from_pretrained(model_path_dit[name])
		base_model = RobertaModel.from_pretrained(model_path_dit[name], output_hidden_states=True)
	return base_model, base_tokenizer


def eval_supervised(data, model, pos_bit=0):
	print(f'Beginning supervised evaluation with {model}...')
	detector = transformers.AutoModelForSequenceClassification.from_pretrained(model, cache_dir=cache_dir).to(DEVICE)
	tokenizer = transformers.AutoTokenizer.from_pretrained(model, cache_dir=cache_dir)

	real, fake = data['original'], data['sampled']
	with torch.no_grad():
		# get predictions for real
		real_preds = []
		for batch in tqdm.tqdm(range(len(real) // batch_size), desc="Evaluating real"):
			batch_real = real[batch * batch_size:(batch + 1) * batch_size]
			batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
			real_preds.extend(detector(**batch_real).logits.softmax(-1)[:,pos_bit].tolist())
		
		# get predictions for fake
		fake_preds = []
		for batch in tqdm.tqdm(range(len(fake) // batch_size), desc="Evaluating fake"):
			batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
			batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
			fake_preds.extend(detector(**batch_fake).logits.softmax(-1)[:,pos_bit].tolist())

	predictions = {
		'real': real_preds,
		'samples': fake_preds,
	}

	fpr, tpr, roc_auc = get_roc_metrics(real_preds, fake_preds)
	p, r, pr_auc = get_precision_recall_metrics(real_preds, fake_preds)
	print(f"{model} ROC AUC: {roc_auc}, PR AUC: {pr_auc}")

	# free GPU memory
	del detector
	torch.cuda.empty_cache()

	return {
		'name': model,
		'predictions': predictions,
		'info': {
			'n_samples': n_samples,
		},
		'metrics': {
			'roc_auc': roc_auc,
			'fpr': fpr,
			'tpr': tpr,
		},
		'pr_metrics': {
			'pr_auc': pr_auc,
			'precision': p,
			'recall': r,
		},
		'loss': 1 - pr_auc,
	}

def fea_get(texts_ls, max_length=300,print_fea_dim=True):

	# tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/DialogRPT-updown")
	# model = transformers.GPT2ForSequenceClassification.from_pretrained("microsoft/DialogRPT-updown")

	with torch.no_grad():
		real_feas = []
		for batch in range(math.ceil(len(texts_ls) / batch_size)):

			batch_real = texts_ls[batch * batch_size : min((batch + 1) * batch_size, len(texts_ls))]
			inputs = base_tokenizer(batch_real, padding='max_length', truncation=True, max_length=max_length,return_tensors="pt").to(DEVICE)
			
			if args.base_model_name not in ['roberta-base', 'roberta-base-openai-detector', 'Hello-SimpleAI/chatgpt-detector-roberta']:
				if args.mask_flag and not args.test_flag:
					input_ids = inputs['input_ids']
					mask_ratio = 0.15
					mask_positions = (torch.rand_like(input_ids.float()) < mask_ratio) & (input_ids != base_tokenizer.cls_token_id) & (input_ids != base_tokenizer.sep_token_id)
					mask_token_id = base_tokenizer.mask_token_id
					input_ids[mask_positions] = mask_token_id
					inputs = {"input_ids": input_ids, "attention_mask": inputs['attention_mask']}
					hidden_states = base_model.transformer(**inputs)[0]
			else:
				outputs = base_model(**inputs)
				hidden_states_all = outputs[2]
				hidden_states = hidden_states_all[-1]

			token_mask_10 = inputs['attention_mask'].unsqueeze(-1)
			hidden_states_mask_10 = hidden_states * token_mask_10

			real_feas.append(hidden_states_mask_10.to('cpu'))
		real_feas_tensor = torch.cat(real_feas,dim=0)
		if print_fea_dim:
			print("Feature dim:", real_feas_tensor.shape)

	return real_feas_tensor

def avg_auroc(all_auroc_list):
	if len(all_auroc_list) <= 2:
		return 0.0, 0.0
	
	# Find the index of the maximum and minimum values
	# max_index = all_auroc_list.index(max(all_auroc_list))
	# min_index = all_auroc_list.index(min(all_auroc_list))
	
	# # Remove the maximum and minimum values
	# filtered_list = [auroc for i, auroc in enumerate(all_auroc_list) if i != max_index and i != min_index]
	filtered_list = all_auroc_list
	
	# Calculate the mean of the remaining values
	avg_auroc = np.round(sum(filtered_list) / len(filtered_list),6)
	
	# Calculate the standard deviation of the remaining values
	std_auroc = np.round(np.std(filtered_list),6)

	return avg_auroc, std_auroc

if __name__ == '__main__':

	DEVICE = "cuda:0"

	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=0)
	parser.add_argument('--dataset', type=str, default="meta_HC3", help='HC3|SQuAD1|NarrativeQA|xsum|TruthfulQA')
	parser.add_argument('--target_datasets', type=str, nargs='+', default="HC3", help='HC3|SQuAD1|NarrativeQA|xsum|TruthfulQA, writing|squad')	
	parser.add_argument('--ref_dataset', type=str, default="", help='HC3|SQuAD1|NarrativeQA|xsum')	
	parser.add_argument('--dataset_key', type=str, default="document")
	parser.add_argument('--pct_words_masked', type=float, default=0.3) # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
	parser.add_argument('--span_length', type=int, default=2)

	parser.add_argument('--n_perturbation_list', type=str, default="1,10")
	parser.add_argument('--n_perturbation_rounds', type=int, default=1)
	parser.add_argument('--base_model_name', type=str, default="roberta-base-openai-detector", help='gpt2-medium|roberta-base|roberta-base-openai-detector')
	parser.add_argument('--text_generated_model_name', type=str, nargs='+', default="gpt2", help='gpt2-medium|roberta-base')
	parser.add_argument('--scoring_model_name', type=str, nargs='+', default="gpt2", help='gpt2-medium|roberta-base')
	parser.add_argument('--mask_filling_model_name', type=str, default="t5-large")
	parser.add_argument('--batch_size', type=int, default=50)
	parser.add_argument('--chunk_size', type=int, default=20, help='number of perturbed texts in tqdm function')
	parser.add_argument('--n_similarity_samples', type=int, default=20)
	parser.add_argument('--int8', action='store_true')
	parser.add_argument('--half', action='store_true')
	parser.add_argument('--base_half', action='store_true')
	parser.add_argument('--do_top_k', action='store_true')
	parser.add_argument('--top_k', type=int, default=40)

	parser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.0002)
	parser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.001)
	parser.add_argument('--update_step', type=int, help='task-level inner update steps', default=10)
	parser.add_argument('--meta_epochs', type=int, default=300)
	parser.add_argument('--n_way', type=int, default=1)
	parser.add_argument('--meta_bs', type=int, default=10)
	parser.add_argument('--k_shot', type=int, default=200)
	parser.add_argument('--k_query', type=int, default=200)

	parser.add_argument('--all_par_num', type=int, default=5000)
	parser.add_argument('--n_samples', type=int, default=1500)
	parser.add_argument('--train_real_num', type=int, default=500)
	parser.add_argument('--target_senten_num', type=int, default=1000)
	parser.add_argument('--train_real_length', type=int, default=0)
	parser.add_argument('--val_num', type=int, default=100)	

	parser.add_argument('--reference_par_num', type=int, default=500)
	parser.add_argument('--test_par_num', type=int, default=1000)
	parser.add_argument('--epochs', type=int, default=50)

	parser.add_argument("--id_load", type=int, help="number of experiment")
	parser.add_argument('--meta_sigma_use', action='store_true')
	parser.add_argument('--no_meta_flag', action='store_true')
	parser.add_argument('--train_LN_flag', action='store_true')
	parser.add_argument('--meta_test_flag', action='store_true')
	parser.add_argument('--meta_best_model_flag', action='store_true')
	parser.add_argument('--test_flag', action='store_true')
	parser.add_argument('--one_senten_flag', action='store_true')
	parser.add_argument('--one_par_flag', action='store_true')
	parser.add_argument('--senten_par_flag', action='store_true')
	
	parser.add_argument('--two_sample_test', action='store_true')
	parser.add_argument('--mask_flag', action='store_true')
	parser.add_argument('--full_data', action='store_true')
	parser.add_argument('--num_mlp', type=int, default=0)
	parser.add_argument('--transformer_flag', action='store_true')	
	parser.add_argument('--num_hidden_layers', type=int, default=1)			
	parser.add_argument('--train_batch_size', type=int, default=200)
	parser.add_argument('--max_length', type=int, default=100)	
	parser.add_argument("--id", type=int, default=998, help="number of experiment")
	parser.add_argument("--epsilon", type=int, default=10, help="10 for imagenet")
	parser.add_argument('--lr', default=0.0001, type=float)	
	parser.add_argument('--sigma0', default=45, type=float, help="0.5 for imagenet")
	parser.add_argument('--sigma', default=30, type=float, help="100 for imagenet")	
	parser.add_argument('--coeff_xy', default=2, type=float)
	parser.add_argument('--target_mlp_num', type=int, default=2)
	parser.add_argument('--target_mlp_lr', default=0.01, type=float)
	parser.add_argument('--trial_num', type=int, default=10)		

	parser.add_argument('--pretaining', action='store_true')
	parser.add_argument('--is_yy_zero', action='store_true')
	parser.add_argument('--MMDO_flag', action='store_true')		

	parser.add_argument('--do_top_p', action='store_true')
	parser.add_argument('--top_p', type=float, default=0.96)
	parser.add_argument('--output_name', type=str, default="")
	parser.add_argument('--openai_model', type=str, default=None)
	parser.add_argument('--openai_key', type=str)
	parser.add_argument('--baselines_only', action='store_true')
	parser.add_argument('--skip_baselines', action='store_true')
	parser.add_argument('--buffer_size', type=int, default=1)
	parser.add_argument('--mask_top_p', type=float, default=1.0)
	parser.add_argument('--pre_perturb_pct', type=float, default=0.0)
	parser.add_argument('--pre_perturb_span_length', type=int, default=5)
	parser.add_argument('--random_fills', action='store_true')
	parser.add_argument('--random_fills_tokens', action='store_true')
	parser.add_argument('--cache_dir', type=str, default="./~/.cache")
	args = parser.parse_args()

	PATH_exper = 'two_sample_test'

	model_path = f'./{PATH_exper}/HC3-{args.base_model_name}/{args.id}'

	if not os.path.isdir(model_path):
		os.makedirs(model_path, exist_ok=True)
	sys.stdout = open(model_path+"/log.log","a")
	print(args)
	current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print("Time Start:", current_time_str)

	API_TOKEN_COUNTER = 0
	all_aruoc_list =[]
	all_power_list = []
	if 'xsum' in args.target_datasets:
		args.n_samples = args.n_samples + 200
	for seed in range(990, 990+args.trial_num):
		current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		print(f"Time Start {seed}:", current_time_str)
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		torch.backends.cudnn.benchmark = False
		torch.backends.cudnn.deterministic = True

		if args.train_batch_size >= args.target_senten_num:
			args.train_batch_size = args.target_senten_num
		
		print("You are Testing!") if args.test_flag else print("You are Traning!")
		if args.openai_model is not None:
			assert args.openai_key is not None, "Must provide OpenAI API key as --openai_key"
			openai.api_key = args.openai_key

		START_DATE = datetime.datetime.now().strftime('%Y-%m-%d')
		START_TIME = datetime.datetime.now().strftime('%H-%M-%S-%f')

		# define SAVE_FOLDER as the timestamp - base model name - mask filling model name
		# create it if it doesn't exist
		precision_string = "int8" if args.int8 else ("fp16" if args.half else "fp32")
		sampling_string = "top_k" if args.do_top_k else ("top_p" if args.do_top_p else "temp")
		output_subfolder = f"{args.output_name}/" if args.output_name else ""
		if args.openai_model is None:
			base_model_name = args.base_model_name.replace('/', '_')
		else:
			base_model_name = "openai-" + args.openai_model.replace('/', '_')
		scoring_model_string = (f"-{args.scoring_model_name}" if args.scoring_model_name else "").replace('/', '_')
		SAVE_FOLDER = f"tmp_results/{output_subfolder}{base_model_name}{scoring_model_string}-{args.mask_filling_model_name}-{sampling_string}/{START_DATE}-{START_TIME}-{precision_string}-{args.pct_words_masked}-{args.n_perturbation_rounds}-{args.dataset}-{args.n_samples}"
		if not os.path.exists(SAVE_FOLDER):
			os.makedirs(SAVE_FOLDER)
		print(f"Saving results to absolute path: {os.path.abspath(SAVE_FOLDER)}")

		# write args to file
		with open(os.path.join(SAVE_FOLDER, "args.json"), "w") as f:
			json.dump(args.__dict__, f, indent=4)

		mask_filling_model_name = args.mask_filling_model_name
		n_samples = args.n_samples
		batch_size = args.batch_size
		n_perturbation_list = [int(x) for x in args.n_perturbation_list.split(",")]
		n_perturbation_rounds = args.n_perturbation_rounds
		n_similarity_samples = args.n_similarity_samples

		cache_dir = args.cache_dir
		os.environ["XDG_CACHE_HOME"] = cache_dir
		if not os.path.exists(cache_dir):
			os.makedirs(cache_dir)
		print(f"Using cache dir {cache_dir}")

		GPT2_TOKENIZER = transformers.GPT2Tokenizer.from_pretrained(model_path_dit['gpt2'])


		# mask filling t5 model
		if not args.baselines_only and not args.random_fills:
			int8_kwargs = {}
			half_kwargs = {}
			if args.int8:
				int8_kwargs = dict(load_in_8bit=True, device_map='auto', torch_dtype=torch.bfloat16)
			elif args.half:
				half_kwargs = dict(torch_dtype=torch.bfloat16)
			print(f'Loading mask filling model {mask_filling_model_name}...')
			mask_model = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_path_dit[mask_filling_model_name], **int8_kwargs, **half_kwargs)
			try:
				n_positions = mask_model.config.n_positions
			except AttributeError:
				n_positions = 512
		else:
			n_positions = 512
		preproc_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_dit['t5-small'], model_max_length=512)
		mask_tokenizer = transformers.AutoTokenizer.from_pretrained(model_path_dit[mask_filling_model_name], model_max_length=n_positions)
		if args.dataset in ['english', 'german']:
			preproc_tokenizer = mask_tokenizer
		
		# Target Dataset
		fea_train_real_ls = []
		fea_train_generated_ls = []
		fea_reference_ls = []
		fea_real_ls = []
		fea_generated_ls = []
		val_real_ls = []
		val_generated_ls =[]

		val_sing_real_ls = []
		val_sing_generated_ls =[]
		for target_dataset in args.target_datasets:
			print(f'Loading dataset {target_dataset}...')
			if target_dataset in ['xsum', 'squad', 'writing']:
				dataset_key = "document" if target_dataset!='squad' else 'context'
				# load data
				if target_dataset in custom_datasets.DATASETS:
					train_real_o = custom_datasets.load(target_dataset, cache_dir)
				else:
					train_real_o = datasets.load_dataset(target_dataset, split='train', cache_dir=cache_dir)[dataset_key]

			elif target_dataset=='HC3':
				data_o = dataset_loader.load(target_dataset, args.cache_dir) # train 38154;   test 9538
				train_real_o = [text for text, label in zip(data_o['train']['text'], data_o['train']['label']) if label == 0] # 19077[text,text]

			else:
				False, f"Do not surpport on {target_dataset}!"
			
			for text_generated_model_name in args.text_generated_model_name:
				
				if text_generated_model_name in ['EleutherAI/gpt-neo-2.7B']:
					batch_size = 20

				elif text_generated_model_name in ['nomic-ai/gpt4all-j','EleutherAI/gpt-j-6b']:
					batch_size = 8

				elif text_generated_model_name in ['EleutherAI/gpt-neo-125m']:
					batch_size = 25
					n_samples = args.n_samples + 300
				else:
					batch_size = args.batch_size
					n_samples = args.n_samples

				if text_generated_model_name=='chatGPT':
					assert target_dataset=='HC3'

				if text_generated_model_name!='chatGPT':
					base_model, base_tokenizer = load_base_model_and_tokenizer(text_generated_model_name)
					load_base_model()

				if target_dataset=='HC3':
					if text_generated_model_name!='chatGPT':
						data = generate_fake_and_combine_real(target_dataset, train_real_o) # data['original'] 2500 
					else:
						data = load_HC3(data_o)
				else:
					data = generate_data(target_dataset, train_real_o)

				if text_generated_model_name!='chatGPT':
					del base_model
					del base_tokenizer
					torch.cuda.empty_cache()
				
				# generic generative model
				base_model, base_tokenizer = load_base_model_and_tokenizer(args.base_model_name)
				load_base_model()

				# training data
				real = data['original']#[:args.train_real_num]  len== n_samples, many sentences of words
				generated = data['sampled']#[:args.train_real_num]
				if args.two_sample_test:
					real_sent_token = [nltk.sent_tokenize(text)[1:-1] for text in real ]
					generated_sent_token = [nltk.sent_tokenize(text)[1:-1] for text in generated ]
					real = [text for text in real_sent_token if len(text)>0]
					generated = [text for text in generated_sent_token if len(text)>0]
					train_length =args.target_senten_num # 200
					train_real_length = args.target_senten_num # 200
					if args.train_real_length>args.target_senten_num:
						train_real_length = args.train_real_length
					train_real = [sen for pa in real[:train_real_length] for sen in pa if 5<len(sen.split())] 
					train_generated =[sen for pa in generated[:train_length] for sen in pa if 5<len(sen.split())]

					real_data = real
					generated_data = generated

					real_data_temp = [[sentence for sentence in sublist if len(sentence.split()) >= 5] for sublist in real_data[train_real_length:]]
					generated_data_temp = [[sentence for sentence in sublist if len(sentence.split()) >= 5] for sublist in generated_data[train_length:]]

					real_data_temp_seletced = [ pa_ls for pa_ls in real_data_temp if len(pa_ls)>=5]
					generated_data_temp_seletced = [ pa_ls for pa_ls in generated_data_temp if len(pa_ls)>=5]
					len_data = min(len(real_data_temp_seletced), len(generated_data_temp_seletced))

					test_lenth = 100
					assert len_data>=args.val_num + test_lenth +250, print(f'Please reduce the args.target_senten_num:{args.target_senten_num}')

				else:
					False, f"You should choose the way to construct the samples!"

				if args.test_flag:
					train_real = train_real[:500]
					train_generated = train_generated[:500]

				fea_train_real = fea_get(train_real, max_length=args.max_length)#.to('cpu')
				fea_train_generated = fea_get(train_generated, max_length=args.max_length)#.to('cpu')

				fea_train_real_ls.append(fea_train_real)
				fea_train_generated_ls.append(fea_train_generated)

				test_lenth = test_lenth

				text_val_real = real_data_temp_seletced[:args.val_num]
				text_val_generated = generated_data_temp_seletced[:args.val_num]
				text_real = real_data_temp_seletced[args.val_num : args.val_num + test_lenth]
				text_generated = generated_data_temp_seletced[args.val_num: args.val_num +test_lenth]

				text_single_real = real_data_temp_seletced[args.val_num + test_lenth : args.val_num + test_lenth + 150]
				text_single_generated = generated_data_temp_seletced[args.val_num +test_lenth : args.val_num +test_lenth + 150]
				text_reference = real_data_temp_seletced[args.val_num + test_lenth + 150: args.val_num + test_lenth +250]

				text_single_real_sen_ls = [ sen for pa in text_single_real for sen in pa]
				text_single_generated_sen_ls = [ sen for pa in text_single_generated for sen in pa]
				text_reference = [ sen for pa in text_reference for sen in pa]


				fea_real = [fea_get(pa_ls, max_length=args.max_length,print_fea_dim=False) for pa_ls in text_real]
				fea_generated = [fea_get(pa_ls, max_length=args.max_length,print_fea_dim=False) for pa_ls in text_generated]

				val_real = [fea_get(pa_ls, max_length=args.max_length,print_fea_dim=False) for pa_ls in text_val_real]
				val_generated = [fea_get(pa_ls, max_length=args.max_length,print_fea_dim=False) for pa_ls in text_val_generated]

				fea_real_ls.extend(fea_real)
				fea_generated_ls.extend(fea_generated)
				val_real_ls.extend(val_real)
				val_generated_ls.extend(val_generated)

				fea_reference = fea_get(text_reference, max_length=args.max_length)
				val_singe_real = fea_get(text_single_real_sen_ls, max_length=args.max_length)
				val_singe_generated = fea_get(text_single_generated_sen_ls, max_length=args.max_length)

				fea_reference_ls.append(fea_reference)
				val_sing_real_ls.append(val_singe_real)
				val_sing_generated_ls.append(val_singe_generated)

		fea_train_real0 = torch.cat(fea_train_real_ls,dim=0)
		fea_train_real0 = fea_train_real0[np.random.permutation(fea_train_real0.shape[0])]
		fea_train_generated0 = torch.cat(fea_train_generated_ls,dim=0)
		fea_train_generated0 = fea_train_generated0[np.random.permutation(fea_train_generated0.shape[0])]
		fea_reference = torch.cat(fea_reference_ls,dim=0)[:500]
		fea_real =  random.sample(fea_real_ls, len(fea_real_ls))
		fea_generated = random.sample(fea_generated_ls, len(fea_generated_ls))
		val_real =  random.sample(val_real_ls, len(val_real_ls))
		val_generated =  random.sample(val_generated_ls, len(val_generated_ls))

		val_sing_real = torch.cat(val_sing_real_ls,dim=0)
		val_sing_generated = torch.cat(val_sing_generated_ls,dim=0)
		val_sing_real = val_sing_real[np.random.permutation(val_sing_real.shape[0])][:1000]
		val_sing_generated = val_sing_generated[np.random.permutation(val_sing_generated.shape[0])][:1000]

		del fea_train_real_ls
		del fea_train_generated_ls
		del fea_reference_ls
		del fea_real_ls
		del fea_generated_ls
		del val_real_ls
		del val_generated_ls
		print("fea_train_real:", fea_train_real0.shape)
		print("fea_train_generated:", fea_train_generated0.shape)
		print("fea_reference:", fea_reference.shape)
		print("fea_real:", len(fea_real))
		print("fea_generated:", len(fea_generated))
		print("val_real:", len(val_real))
		print("val_generated:", len(val_generated))
		print("val_sing_real:", len(val_sing_real))
		print("val_sing_generated:", len(val_sing_generated))

		train_batch_size = args.train_batch_size
		auroc_list = []
		auroc_value_best = 0
		power_best = 0
		id = args.id
		def test(epoch, test_lenth=2000, fea_real = fea_real, fea_generated = fea_generated, meta_save_model_flag='', test_flag = args.test_flag):
			net.eval()
			global auroc_value_best
			with torch.no_grad():
				feature_ref_ls = []
				if len(fea_reference) > train_batch_size:
					for batch in tqdm.tqdm(range(len(fea_reference) // train_batch_size), desc="Testing for deep MMD"):
						feature_ref = net(fea_reference[batch * train_batch_size:(batch + 1) * train_batch_size].to('cuda'))
						feature_ref_ls.append(feature_ref)
				else:
					feature_ref = net(fea_reference.to('cuda'))
					feature_ref_ls.append(feature_ref)
				feature_ref = torch.cat(feature_ref_ls,dim=0)

				def get_feature_cln_ls(fea_real):
					feature_cln_ls=[]
					if len(fea_real) > train_batch_size:
						for	batch in range(len(fea_real) // train_batch_size):		
							feature_cln = net(fea_real[batch * train_batch_size:(batch + 1) * train_batch_size].to('cuda'))						
							feature_cln_ls.append(feature_cln)
					else:
						feature_cln = net(fea_real.to('cuda'))
						# feature_adv = net(fea_generated.to('cuda'))
						feature_cln_ls.append(feature_cln)
						# feature_adv_ls.append(feature_adv)
					return feature_cln_ls
				
				feature_cln_ls = get_feature_cln_ls(fea_real)
				feature_adv_ls = get_feature_cln_ls(fea_generated)

				feature_cln = torch.cat(feature_cln_ls,dim=0)
				feature_adv = torch.cat(feature_adv_ls,dim=0)

				dt_clean = MMD_batch2(torch.cat([feature_ref,feature_cln],dim=0), feature_ref.shape[0], torch.cat([fea_reference[:feature_ref.shape[0]].to('cuda'),fea_real[:feature_cln.shape[0]].to('cuda')],dim=0).view(feature_ref.shape[0]+feature_cln.shape[0],-1), sigma, sigma0_u, ep).to('cpu')
				dt_adv = MMD_batch2(torch.cat([feature_ref,feature_adv],dim=0), feature_ref.shape[0], torch.cat([fea_reference[:feature_ref.shape[0]].to('cuda'),fea_generated[:feature_adv.shape[0]].to('cuda')],dim=0).view(feature_ref.shape[0]+feature_adv.shape[0],-1), sigma, sigma0_u, ep).to('cpu')
				auroc_value = plot_mi(dt_clean, dt_adv)
				auroc_list.append(auroc_value)
				model_path = f'./{PATH_exper}/HC3-{args.base_model_name}/{id}' 

				state = {
						'net': net.state_dict(),
						# 'epsilonOPT': epsilonOPT,
						# 'sigmaOPT': sigmaOPT,
						# 'sigma0OPT': sigma0OPT,
						'sigma': sigma,
						'sigma0_u':sigma0_u,
						'ep': ep
					}
				
				if not test_flag:
					if not os.path.isdir(model_path):
						os.makedirs(model_path, exist_ok=True)
						# os.mkdir(model_path)
					# if (epoch+1)%100==0:
					# 	torch.save(state, model_path + '/'+ str(epoch) +'_ckpt.pth')
					if auroc_value> auroc_value_best:
						auroc_value_best = auroc_value
						torch.save(state, model_path + '/'+ meta_save_model_flag +'best_ckpt.pth')
						print("Save the best model!")
				# torch.save(state, model_path + '/'+ meta_save_model_flag +'last_ckpt.pth')
			return auroc_value
		
		def two_sample_test(epoch, test_lenth=2000, fea_real_ls = fea_real, fea_generated_ls = fea_generated, meta_save_model_flag='', test_flag = args.test_flag,N=100):
			global power_best
			net.eval()
			fea_real_ls = fea_real_ls[:min(len(fea_real_ls), len(fea_generated_ls))]
			fea_generated_ls = fea_generated_ls[:min(len(fea_real_ls), len(fea_generated_ls))]
			with torch.no_grad():

				def mmd_two_sampe(fea_real_ls, fea_generated_ls,N=100):
					fea_real_ls = fea_real_ls[:min(len(fea_real_ls), len(fea_generated_ls))]
					fea_generated_ls = fea_generated_ls[:min(len(fea_real_ls), len(fea_generated_ls))]

					test_power_ls = []

					N_per = 50
					alpha = 0.05
					for i in range(len(fea_real_ls)):

						# if i>100:
						# 	break
						fea_x_ori = fea_real_ls[i].to('cuda')
						fea_y_ori = fea_generated_ls[i].to('cuda')
						fea_x_ori = fea_x_ori[:min(len(fea_x_ori), len(fea_y_ori))]
						fea_y_ori = fea_y_ori[:min(len(fea_x_ori), len(fea_y_ori))]
						final_x = net(fea_x_ori)
						final_y = net(fea_y_ori)
						count_u = 0
						for _ in range(N):
							h_u, threshold_u, mmd_value_u = TST_MMD_u(torch.cat([final_x,final_y],dim=0), N_per, final_x.shape[0], torch.cat([fea_x_ori, fea_y_ori],dim=0).view(fea_x_ori.shape[0]+fea_y_ori.shape[0],-1), sigma, sigma0_u, ep, alpha)
							count_u = count_u + h_u

						test_power_ls.append(count_u/N)

					return test_power_ls
				generated_test_power_ls = mmd_two_sampe(fea_real_ls, fea_generated_ls, N=N)
				power = sum(generated_test_power_ls)/len(generated_test_power_ls)
				print(f"Test power: {np.round(power,6)}")
				model_path = f'./{PATH_exper}/HC3-{args.base_model_name}/{id}' 


				state = {
						'net': net.state_dict(),
						# 'epsilonOPT': epsilonOPT,
						# 'sigmaOPT': sigmaOPT,
						# 'sigma0OPT': sigma0OPT,
						'sigma': sigma,
						'sigma0_u':sigma0_u,
						'ep': ep
					}
				
				if not test_flag:
					if not os.path.isdir(model_path):
						os.makedirs(model_path, exist_ok=True)
						# os.mkdir(model_path)
					# if (epoch+1)%100==0:
					# 	torch.save(state, model_path + '/'+ str(epoch) +'_ckpt.pth')
					if power> power_best:
						power_best = power
						torch.save(state, model_path + '/'+ meta_save_model_flag +'best_ckpt.pth')
						print("Save the best model!")
				# torch.save(state, model_path + '/'+ meta_save_model_flag +'last_ckpt.pth')
			return power
		
		_, token_num, hidden_size = fea_train_real0.size()
		fea_dim = token_num * hidden_size
		Config = namedtuple('Config', ['in_dim', 'hid_dim', 'dropout', 'out_dim', 'token_num'])
		config = Config(
				in_dim=hidden_size,
				token_num=token_num,
				hid_dim=512,
				dropout=0.2,
				out_dim=300,)
		
		# train
		device = torch.device("cuda:0")
		maml = Meta(args, config).to(device)
		
		id = args.id
		net = mmdPreModel(config=config, num_mlp=args.num_mlp, transformer_flag=args.transformer_flag, num_hidden_layers=args.num_hidden_layers).cuda()
		print('==> loading meta_model from checkpoint..')
		# model_path = f'./net_D/resnet101/{id}' 
		print("No meta learing!")
		sigma, sigma0_u, ep  = maml.sigmaOPT ** 2, maml.sigma0OPT ** 2, maml.epsilonOPT ** 2
		if args.MMDO_flag:
			ep = torch.ones(1).to('cuda', torch.float)
		print('==> testing from the loaded checkpoint..')
		num_target = len(fea_real)//test_lenth

		power_ls = []
		for i in range(num_target):
			power = two_sample_test(0, fea_real_ls=fea_real[i*test_lenth:(i+1)*test_lenth], fea_generated_ls=fea_generated[i*test_lenth:(i+1)*test_lenth],test_flag = True,N=10)
			power_ls.append(power)
		print("average power_value:", sum(power_ls)/len(power_ls))

		device = 'cuda' if torch.cuda.is_available() else 'cpu'
		# Initialize parameters
		epsilonOPT = torch.from_numpy(np.ones(1) * np.sqrt(ep.detach().cpu().numpy())).to(device, torch.float)
		epsilonOPT.requires_grad = True
		sigmaOPT = torch.from_numpy(np.ones(1) * np.sqrt(sigma.detach().cpu().numpy())).to(device, torch.float)
		sigmaOPT.requires_grad = True
		sigma0OPT = torch.from_numpy(np.ones(1) * np.sqrt(sigma0_u.detach().cpu().numpy())).to(device, torch.float)
		sigma0OPT.requires_grad = True

		sigma, sigma0_u, ep = None, None, None
		optimizer = torch.optim.Adam(list(net.parameters())+ [epsilonOPT] + [sigmaOPT] + [sigma0OPT], lr=args.lr)
		epochs = args.epochs
		train_batch_size = args.train_batch_size

		def train(epoch):
			print('\nEpoch: %d' % epoch)
			
			net.train()
			for batch in tqdm.tqdm(range(len(fea_train_generated) // train_batch_size), desc="Traning for deep MMD"):
				inputs = fea_train_real[batch * train_batch_size:(batch + 1) * train_batch_size]
				x_adv = fea_train_generated[batch * train_batch_size:(batch + 1) * train_batch_size]

				if inputs.shape[0]!=x_adv.shape[0]:
					break
				inputs = inputs.cuda(non_blocking=True)
				x_adv = x_adv.cuda(non_blocking=True)
				assert inputs.shape[0]==x_adv.shape[0]

				X = torch.cat([inputs, x_adv],dim=0)

				optimizer.zero_grad()
				outputs = net(X)
				
				ep = epsilonOPT **2
				sigma = sigmaOPT ** 2
				sigma0_u = sigma0OPT ** 2
				# Compute Compute J (STAT_u)
				TEMP = MMDu(outputs, inputs.shape[0], X.view(X.shape[0],-1), sigma, sigma0_u, ep, coeff_xy=args.coeff_xy, is_yy_zero=args.is_yy_zero)
				mmd_value_temp = -1 * (TEMP[0])
				mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
				STAT_u = torch.div(mmd_value_temp, mmd_std_temp)
				
				# Compute gradient
				STAT_u.backward()

				# Update weights using gradient descent
				optimizer.step()
			
			print(f"epoch:{epoch}, mmd_value_temp:{mmd_value_temp.item()}, STAT_u:{STAT_u.item()}")
			return sigma, sigma0_u, ep

		id = args.id
		start_epoch = 0
		auroc_value_best_epoch = 0
		if not args.test_flag:
			for epoch in range(start_epoch, start_epoch+epochs):
				time0 = time.time()
				
				fea_train_real0 = fea_train_real0[np.random.permutation(fea_train_real0.shape[0])]
				fea_train_generated0 = fea_train_generated0[np.random.permutation(fea_train_generated0.shape[0])]
				if len(fea_train_real0)>=len(fea_train_generated0):
					for i in range(len(fea_train_real0)//len(fea_train_generated0)):
						fea_train_real = fea_train_real0[fea_train_generated0.shape[0] * i:fea_train_generated0.shape[0] * (i+1)]
						fea_train_generated = fea_train_generated0
						sigma, sigma0_u, ep =train(epoch)
				else:
					for i in range(len(fea_train_generated0)//len(fea_train_real0)):
						fea_train_generated = fea_train_generated0[len(fea_train_real0) * i: len(fea_train_real0) * (i+1)]
						fea_train_real = fea_train_real0
						sigma, sigma0_u, ep =train(epoch)
				print("train time:",time.time()-time0)
				time0 = time.time()
				if (epoch+1)%1==0:
					power = two_sample_test(epoch, fea_real_ls= val_real, fea_generated_ls= val_generated,N=10,test_flag = True)
					auroc_value_epoch = test(epoch, fea_real= val_sing_real, fea_generated= val_sing_generated)
					if auroc_value_epoch>auroc_value_best_epoch: auroc_value_best_epoch = auroc_value_epoch 
				print("test time:",time.time()-time0)

			print('==> loading meta_best_model from checkpoint..')
			model_path = f'./{PATH_exper}/HC3-{args.base_model_name}/{id}'
			assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
			checkpoint = torch.load(model_path + '/'+ 'best_ckpt.pth')
			net.load_state_dict(checkpoint['net'])
			sigma, sigma0_u, ep  = checkpoint['sigma'], checkpoint['sigma0_u'], checkpoint['ep']
			print('==> testing from the loaded checkpoint..')
			power = two_sample_test(epoch,test_flag = True)
			auroc_value = test(epoch, fea_real= val_sing_real, fea_generated= val_sing_generated,test_flag = True)
			print('==> testing each model..')
			for i in range(len(args.text_generated_model_name)):
				test(0, fea_real=val_sing_real_ls[i][:1000], fea_generated=val_sing_generated_ls[i][:1000],test_flag = True)
			print(f"{id}'s best power is {power}!")
			print(f"and the corresponding auroc is {auroc_value}!")
			print(f"but the best auroc is {auroc_value_best_epoch}!")

		else:
			epoch = 99
			print('==> testing from checkpoint..')
			model_path = f'./{PATH_exper}/HC3-{args.base_model_name}/{id}'
			if not os.path.isdir(model_path):
				model_path = f'./{PATH_exper}/HC3-gpt2/999'
				print(f"Note you are loading {model_path}")
			assert os.path.isdir(model_path), 'Error: no checkpoint directory found!'
			checkpoint = torch.load(model_path + '/'+ 'best_ckpt.pth')
			net.load_state_dict(checkpoint['net'])
			sigma, sigma0_u, ep  = checkpoint['sigma'], checkpoint['sigma0_u'], checkpoint['ep']
			# test(epoch)
			auroc_value = test(epoch, fea_real= val_sing_real, fea_generated= val_sing_generated,test_flag = True)
			power = two_sample_test(epoch)
		all_power_list.append(np.round(power,6))
		all_aruoc_list.append(np.round(auroc_value,6))
		print(f"The best power list is {all_power_list}!")
		print(f"The best auroc list is {all_aruoc_list}!")
		print(f"avg_power: {avg_auroc(all_power_list)[0]} and std_power: {avg_auroc(all_power_list)[1]}")
		print(f"avg_auroc: {avg_auroc(all_aruoc_list)[0]} and std_auroc: {avg_auroc(all_aruoc_list)[1]}")

	current_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	print("Time Over:", current_time_str)
