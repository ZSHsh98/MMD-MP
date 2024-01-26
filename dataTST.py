import torch
import numpy as np


def get_metadata(args, fea_dic):
	
	meta_bs = args.meta_bs # adjust according to the gpu
	num_meta_sources = len(fea_dic)
	n_way = args.n_way
	k_shot = args.k_shot #// n_way
	k_query = args.k_query #// n_way
	assert (k_shot + k_query) <= 500

	def get_data_num(selected_source):
		return len(fea_dic[selected_source]['real'])
	

	x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
	for i in range(meta_bs):  # one batch means one set
		x_spt, y_spt, x_qry, y_qry = [], [], [], []
		selected_source_real = np.random.choice(num_meta_sources, n_way, False)
		selected_source_generated = np.random.choice(num_meta_sources, n_way, False)

		for cur_source_real, cur_source_generated in zip(selected_source_real, selected_source_generated):
			selected_real = np.random.choice(get_data_num(cur_source_real), (k_shot + k_query), False)
			selected_generated = np.random.choice(get_data_num(cur_source_generated), (k_shot + k_query), False)
			
			x_spt.append(fea_dic[cur_source_real]['real'][selected_real[:k_shot]])
			x_qry.append(fea_dic[cur_source_real]['real'][selected_real[k_shot :]])
			y_spt.append(fea_dic[cur_source_generated]['generated'][selected_generated[:k_shot]])
			y_qry.append(fea_dic[cur_source_generated]['generated'][selected_generated[k_shot :]])

		# shuffle inside a batch
		_, token_num, hidden_size = x_spt[0].size()
		perm = np.random.permutation(n_way * k_shot)
		x_spt = torch.cat(x_spt).reshape(n_way * k_shot, token_num, hidden_size)[perm]
		y_spt = torch.cat(y_spt).reshape(n_way * k_shot, token_num, hidden_size)[perm]
		perm = np.random.permutation(n_way * k_query)
		x_qry = torch.cat(x_qry).reshape(n_way * k_query, token_num, hidden_size)[perm]
		y_qry = torch.cat(y_qry).reshape(n_way * k_query, token_num, hidden_size)[perm]

		# append tasks
		x_spts.append(x_spt)
		y_spts.append(y_spt)
		x_qrys.append(x_qry)
		y_qrys.append(y_qry)
	
	# generate batch
	x_spts = torch.cat(x_spts).reshape(meta_bs, n_way * k_shot, token_num, hidden_size)
	y_spts = torch.cat(y_spts).reshape(meta_bs, n_way * k_shot, token_num, hidden_size)
	x_qrys = torch.cat(x_qrys).reshape(meta_bs, n_way * k_query, token_num, hidden_size)
	y_qrys = torch.cat(y_qrys).reshape(meta_bs, n_way * k_query, token_num, hidden_size)
	
	return x_spts, y_spts, x_qrys, y_qrys