import numpy as np
import torch

def L2_distance_get(x, y, value_factor = 1):
	"""compute the paired distance between x and y."""
	x_norm = ((x/value_factor) ** 2).sum(1).view(-1, 1)
	y_norm = ((y/value_factor) ** 2).sum(1).view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x/value_factor, torch.transpose(y/value_factor, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist

def guassian_kernel(source, target, kernel_mul, kernel_num=10, fix_sigma=None):
	"""计算Gram核矩阵
	source: sample_size_1 * feature_size 的数据
	target: sample_size_2 * feature_size 的数据
	kernel_mul: 这个概念不太清楚,感觉也是为了计算每个核的bandwith
	kernel_num: 表示的是多核的数量
	fix_sigma: 表示是否使用固定的标准差
		return: (sample_size_1 + sample_size_2) * (sample_size_1 + sample_size_2)的
						矩阵，表达形式:
						[   K_ss K_st
							K_ts K_tt ]
	"""
	n_samples = int(source.size()[0])+int(target.size()[0])
	# total = torch.cat([source, target], dim=0) # 合并在一起

	# total0 = total.unsqueeze(0).expand(int(total.size(0)), \
	# 								   int(total.size(0)), \
	# 								   int(total.size(1)))
	# total1 = total.unsqueeze(1).expand(int(total.size(0)), \
	# 								   int(total.size(0)), \
	# 								   int(total.size(1)))
	# value_factor = 1
	# L2_distance = (((total0-total1)/value_factor)**2).sum(2) # 计算高斯核中的|x-y|
	# L2_distance = ((total0-total1)**2).sum(2) # 计算高斯核中的|x-y|
	# assert not torch.isinf(L2_distance).any(), 'tune the value_factor larger'
	L2_distance = L2_distance_get(source, target)

	# 计算多核中每个核的bandwidth
	if fix_sigma:
		bandwidth = fix_sigma
	else:
		bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
		# bandwidth = torch.sum(L2_distance.data / ((n_samples**2-n_samples)/(value_factor)**2) /(value_factor)**2)
		assert not torch.isinf(bandwidth).any()
	bandwidth = bandwidth / (kernel_mul ** (kernel_num // 2))
	scale_factor = 0
	bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
	# bandwidth_list = [torch.clamp(bandwidth * (kernel_mul**scale_factor) *(kernel_mul**(i-scale_factor)),max=1.0e38) for i in range(kernel_num)]

	# 高斯核的公式，exp(-|x-y|/bandwith)
	kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for \
				  bandwidth_temp in bandwidth_list]

	return sum(kernel_val)/kernel_num # 将多个核合并在一起

def mmd(source, target, kernel_mul=2.0, kernel_num=10, fix_sigma=None):
	n = int(source.size()[0])
	m = int(target.size()[0])

	kernels = guassian_kernel(source, target,
							  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
	XX = kernels[:n, :n] 
	YY = kernels[n:, n:]
	XY = kernels[:n, n:]
	YX = kernels[n:, :n]

	XX = torch.div(XX, n * n).sum(dim=1).view(1,-1)  # K_ss矩阵，Source<->Source
	XY = torch.div(XY, -n * m).sum(dim=1).view(1,-1) # K_st矩阵，Source<->Target

	YX = torch.div(YX, -m * n).sum(dim=1).view(1,-1) # K_ts矩阵,Target<->Source
	YY = torch.div(YY, m * m).sum(dim=1).view(1,-1)  # K_tt矩阵,Target<->Target
		
	loss = (XX + XY).sum() + (YX + YY).sum()
	# loss = XY.sum()
	return loss