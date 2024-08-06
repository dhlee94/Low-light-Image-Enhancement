import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Function
import logging

def total_variation_loss(x):
    h_diff = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
    w_diff = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
    loss = (h_diff + w_diff) / (x.size(0) * x.size(1) * x.size(2) * x.size(3))
    return loss

def color_loss(output, target):
    return 1 - torch.mean(torch.cosine_similarity(output, target, dim=1))

class HuberLoss(nn.Module):
    def __init__(self, delta=0.5):
        super(HuberLoss, self).__init__()
        self.delta = delta
    
    def forward(self, y_true, y_pred):
        error = y_true - y_pred
        abs_error = torch.abs(error)
        is_small_error = abs_error <= self.delta
        small_error_loss = 0.5 * error**2
        big_error_loss = self.delta * (abs_error - 0.5 * self.delta)
        loss = torch.where(is_small_error, small_error_loss, big_error_loss)
        return loss.mean()
    
class GroupContrastiveLoss(nn.Module):
	def __init__(self, temperature=0.5):
		super().__init__()
		self.register_buffer("temperature", torch.tensor(temperature))

	def forward(self, emb_i, emb_j):
		"""
		emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
		z_i, z_j as per SimCLR paper
		"""
		batch_size = emb_i.shape[0]
		device = emb_i.get_device()
		if device>=0:
			negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(torch.device(f"cuda:{device}"))
			positives_mask = (~torch.eye(batch_size * 1, batch_size * 1, dtype=bool)).float().to(torch.device(f"cuda:{device}"))
		else:
			negatives_mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(torch.device("cpu"))
			positives_mask = (~torch.eye(batch_size * 1, batch_size * 1, dtype=bool)).float().to(torch.device("cpu"))
		negatives_mask[:len(emb_i), :len(emb_j)]=False
		negatives_mask[len(emb_i):, len(emb_j):] = False

		z_i = F.normalize(emb_i, dim=1)
		z_j = F.normalize(emb_j, dim=1)

		representations = torch.cat([z_i, z_j], dim=0)
		similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

		pos_similarity_matrix = similarity_matrix[:len(emb_i), :len(emb_j)]
		neg_similarity_matrix = similarity_matrix[len(emb_i):, len(emb_j):]

		pos_similarity_matrix = pos_similarity_matrix * positives_mask
		sim_ij=torch.sum(pos_similarity_matrix,dim=1)/(len(neg_similarity_matrix)-1)

		neg_similarity_matrix = neg_similarity_matrix * positives_mask
		sim_ji = torch.sum(neg_similarity_matrix, dim=1)/(len(neg_similarity_matrix)-1)

		positives = torch.cat([sim_ij, sim_ji], dim=0)

		nominator = torch.exp(positives / self.temperature)
		denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)

		loss_partial = torch.sum(nominator / (nominator + torch.sum(denominator, dim=1)))/ (2 * batch_size)
		loss = -torch.log(loss_partial)

		return loss

def calculate_rank_loss(features, high_distorted_features, low_distorted_features):
    d_high = torch.norm(features - high_distorted_features, dim=1)
    d_low = torch.norm(features - low_distorted_features, dim=1)
    
    probabilities = torch.sigmoid(d_high - d_low)
    target = torch.ones_like(probabilities)
    
    loss = F.binary_cross_entropy(probabilities, target)
    return loss