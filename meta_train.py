import  torch
from    torch import nn
from utils_MMD import MMDu
import numpy as np
import math
dtype = torch.float
device = torch.device("cuda:0")
from copy import deepcopy
from pytorch_transformers.modeling_bert import(
	BertEncoder,
	BertPreTrainedModel,
	BertConfig
)

class GeLU(nn.Module):
	"""Implementation of the gelu activation function.
		For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
		0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
		Also see https://arxiv.org/abs/1606.08415
	"""

	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BertLayerNorm(nn.Module):
	def __init__(self, hidden_size, eps=1e-12):
		"""Construct a layernorm module in the TF style (epsilon inside the square root).
		"""
		super(BertLayerNorm, self).__init__()
		self.weight = nn.Parameter(torch.ones(hidden_size))
		self.bias = nn.Parameter(torch.zeros(hidden_size))
		self.variance_epsilon = eps

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.weight * x + self.bias

class mlp_meta(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.mlp = nn.Sequential(
			nn.Linear(config.hid_dim, config.hid_dim),
			GeLU(),
			BertLayerNorm(config.hid_dim, eps=1e-12),
			nn.Dropout(config.dropout),
		)

	def forward(self, x):
		return self.mlp(x)
	
class Bert_Transformer_Layer(BertPreTrainedModel):
	def __init__(self,fusion_config):
		super().__init__(BertConfig(**fusion_config))
		bertconfig_fusion = BertConfig(**fusion_config)
		self.encoder = BertEncoder(bertconfig_fusion)
		self.init_weights()
	
	def forward(self,input, mask=None):
		"""
		input:(bs, 4, dim)
		"""
		batch, feats, dim = input.size()
		if mask is not None:
			mask_ = torch.ones(size=(batch,feats), device=mask.device)
			mask_[:,1:] = mask
			mask_ = torch.bmm(mask_.view(batch,1,-1).transpose(1,2), mask_.view(batch,1,-1))
			mask_ = mask_.unsqueeze(1)
		
		else:
			mask = torch.Tensor([1.0]).to(input.device)
			mask_ = mask.repeat(batch,1,feats, feats)
		
		extend_mask = (1- mask_) * -10000
		assert not extend_mask.requires_grad
		head_mask = [None] * self.config.num_hidden_layers

		enc_output = self.encoder(
			input,extend_mask,head_mask=head_mask
		)
		output = enc_output[0]
		all_attention = enc_output[1]

		return output,all_attention
	
class mmdPreModel(nn.Module):
	def __init__(self, config, num_mlp=0, transformer_flag=False, num_hidden_layers=1, mlp_flag=True):
		super(mmdPreModel, self).__init__()
		self.num_mlp = num_mlp
		self.transformer_flag = transformer_flag
		self.mlp_flag = mlp_flag
		token_num = config.token_num
		self.mlp = nn.Sequential(
			nn.Linear(config.in_dim, config.hid_dim),
			GeLU(),
			BertLayerNorm(config.hid_dim, eps=1e-12),
			nn.Dropout(config.dropout),
			# nn.Linear(config.hid_dim, config.out_dim),
		)
		self.fusion_config = {
			'hidden_size': config.in_dim,
			'num_hidden_layers':num_hidden_layers,
			'num_attention_heads':4,
			'output_attentions':True
			}
		if self.num_mlp>0:
			self.mlp2 = nn.ModuleList([mlp_meta(config) for _ in range(self.num_mlp)])
		if self.transformer_flag:
			self.transformer = Bert_Transformer_Layer(self.fusion_config)
		self.feature = nn.Linear(config.hid_dim * token_num, config.out_dim)
		# self.linear1 = nn.ModuleList([nn.Linear(a, b) for a,b in zip(config.linear_ls_1[:-1], config.linear_ls_1[1:])])
		# self.linear2 = nn.ModuleList([nn.Linear(a, b) for a,b in zip(config.linear_ls_2[:-1], config.linear_ls_2[1:])])
		# self.feature = nn.Linear(config.linear_ls_2[-1], config.out_dim)
	def forward1(self,features):
		features = features
		if self.transformer_flag:
			features,_ = self.transformer(features)
		# features = self.mlp(features)
		return features
	
	def forward2(self,features):
		features = self.mlp(features)
		if self.num_mlp>0:
			# features = self.mlp2(features)
			for _ in range(1):
				for mlp in self.mlp2:
					features = mlp(features)
		features = self.feature(features.view(features.shape[0], -1))
		return features
	
	def forward(self, features):
		"""
		input: [batch, token_num, hidden_size], output: [batch, token_num * config.out_dim]
		"""

		if self.transformer_flag:
			features,_ = self.transformer(features)
		if self.mlp_flag:
			features = self.mlp(features)
		
		if self.num_mlp>0:
			# features = self.mlp2(features)
			for _ in range(1):
				for mlp in self.mlp2:
					features = mlp(features)

		features = self.feature(features.view(features.shape[0], -1))
		return features #features.view(features.shape[0], -1)

	
class Meta(nn.Module):
	"""
	Meta Learner
	"""
	def __init__(self, args, config):

		super(Meta, self).__init__()

		self.update_lr = args.update_lr
		self.meta_lr = args.meta_lr
		self.update_step = args.update_step
		self.train_LN_flag = args.train_LN_flag
		self.coeff_xy=args.coeff_xy

		# Model
		self.net = mmdPreModel(config=config, num_mlp=args.num_mlp, transformer_flag=args.transformer_flag, num_hidden_layers=args.num_hidden_layers).cuda()
		# self.net_temp = mmdPreModel(config=config, num_mlp=args.num_mlp, transformer_flag=args.transformer_flag, num_hidden_layers=args.num_hidden_layers).cuda()

		self.fea_dim = config.in_dim * config.token_num
		self.epsilonOPT = torch.from_numpy(np.random.rand(1) * 10 ** (-args.epsilon)).to(device, torch.float)
		self.epsilonOPT.requires_grad = True
		self.sigmaOPT = torch.from_numpy(np.ones(1) * np.sqrt(2 * self.fea_dim*args.sigma)).to(device, torch.float)
		self.sigmaOPT.requires_grad = True
		self.sigma0OPT = torch.from_numpy(np.ones(1) * np.sqrt(args.sigma0)).to(device, torch.float)
		self.sigma0OPT.requires_grad = True
		self.meta_optim = torch.optim.Adam(list(self.net.parameters())+ [self.epsilonOPT] + [self.sigmaOPT] + [self.sigma0OPT], lr=self.meta_lr)
		self.net.train()
	def forward(self, x_spt, y_spt, x_qry, y_qry, is_training = True):
		"""

		:param x_spt:   [b, setsz, d]
		:param y_spt:   [b, setsz, d]
		:param x_qry:   [b, querysz, d]
		:param y_qry:   [b, querysz, d]
		:return:
		"""
		# task_num, setsz, d = x_spt.size()

		if self.train_LN_flag:
			self.net.train()

		task_num = x_spt.size(0)
		setsz = x_spt.size(1)
		querysz = x_qry.size(1)

		losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i

		loss_not_train = torch.tensor(0.0)


		sum_gradients = []
		for i in range(task_num):
			# Get two samples
			S_spt = torch.cat((x_spt[i],y_spt[i]), 0).to(device, dtype) # 300 *2
			S_qry = torch.cat((x_qry[i], y_qry[i]), 0).to(device, dtype) # 100 *2
			self.net_temp = deepcopy(self.net).cuda()
			self.net_temp.train()

			# Initialize parameters
			ep = self.epsilonOPT ** 2
			sigma = self.sigmaOPT ** 2
			sigma0_u = self.sigma0OPT ** 2

			if is_training == False:
				return loss_not_train, self.net, sigma, sigma0_u, ep

			model_output = self.net(S_qry)
			TEMP = MMDu(model_output, querysz, S_qry.view(S_qry.shape[0], -1), sigma, sigma0_u, ep, ep,coeff_xy=self.coeff_xy)
			mmd_value_temp = -1 * TEMP[0]
			mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
			loss = torch.div(mmd_value_temp, mmd_std_temp)
			grad = torch.autograd.grad(loss, self.net.parameters())
			for param, grad in zip(self.net_temp.parameters(), grad):
				param.data = param.data - self.update_lr * grad

			# # first update
			# with torch.no_grad():
			# 	model_output_q = self.net(S_spt)
			# 	TEMP = MMDu(model_output_q, setsz, S_spt, sigma, sigma0_u, ep)
			# 	mmd_value_temp = -1 * TEMP[0]
			# 	mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
			# 	loss_q = torch.div(mmd_value_temp, mmd_std_temp)
			# 	losses_q[0] += loss_q

			# # second update
			# with torch.no_grad():
			# 	model_output_q = self.net_temp(S_spt)
			# 	TEMP = MMDu(model_output_q, setsz, S_spt, sigma, sigma0_u, ep)
			# 	mmd_value_temp = -1 * TEMP[0]
			# 	mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
			# 	loss_q = torch.div(mmd_value_temp, mmd_std_temp)
			# 	losses_q[1] += loss_q

			for k in range(1, self.update_step):
				# run the i-th task and compute loss for k=1~K-1
				model_output = self.net_temp(S_qry)
				TEMP = MMDu(model_output, querysz, S_qry.view(S_qry.shape[0], -1), sigma, sigma0_u, ep)
				mmd_value_temp = -1 * TEMP[0]
				mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
				loss = torch.div(mmd_value_temp, mmd_std_temp)
				# loss = F.cross_entropy(logits, y_spt[i])
				grad = torch.autograd.grad(loss, self.net_temp.parameters())
				for param, grad in zip(self.net_temp.parameters(), grad):
					param.data = param.data - self.update_lr * grad

				# record the loss
				# if k<self.update_step-1:
				# 	with torch.no_grad():
				# 		model_output_q = self.net(S_spt, self.net_temp.parameters())
				# 		TEMP = MMDu(model_output_q, setsz, S_spt, sigma, sigma0_u, ep)
				# 		mmd_value_temp = -1 * TEMP[0]
				# 		mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
				# 		loss_q = torch.div(mmd_value_temp, mmd_std_temp)
				# 		# print(loss_q.item())
				# 		losses_q[k + 1] += loss_q
				# else:
			model_output_q = self.net_temp(S_spt)
			TEMP = MMDu(model_output_q, setsz, S_spt.view(S_spt.shape[0], -1), sigma, sigma0_u, ep)
			mmd_value_temp = -1 * TEMP[0]
			mmd_std_temp = torch.sqrt(TEMP[1] + 10 ** (-8))
			loss_q = torch.div(mmd_value_temp, mmd_std_temp)
			# print(loss_q.item())
			losses_q[k + 1] += loss_q
			loss_q.backward()
			for j, params in enumerate(self.net_temp.parameters()):
				if i == 0:
					sum_gradients.append(deepcopy(params.grad))
				else:
					sum_gradients[j] += deepcopy(params.grad)
			
		# print('sigma:',sigma.item(), 'sigma0:',sigma0_u.item(), 'epsilon:',ep.item())
		print(f"mmd_value_temp:{mmd_value_temp.item()}, STAT_u:{loss_q.item()}, mmd_std:{mmd_std_temp.item()}")

		# sum over all losses across all tasks
		loss_q = losses_q[-1] / task_num
		print('J_value:',-loss_q.item())

		# optimize theta parameters
		for i, params in enumerate(self.net.parameters()):
			params.grad = sum_gradients[i] / task_num
		self.epsilonOPT.grad, self.sigmaOPT.grad, self.sigma0OPT.grad = self.epsilonOPT.grad/task_num, self.sigmaOPT.grad/task_num, self.sigma0OPT.grad/task_num
		self.meta_optim.step()
		self.meta_optim.zero_grad()
		del sum_gradients

		return -1 * loss_q, self.net, sigma, sigma0_u, ep
