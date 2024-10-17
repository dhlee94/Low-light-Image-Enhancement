import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from models.network import DualColorNetwork
from models.quality_network import QualityNetwork
from core.criterion import GroupContrastiveLoss
from core.optimizer import CosineAnnealingWarmUpRestarts
from utils.utils import rotate_batch, calculate_quality_diff
from prettytable import PrettyTable
import time

class Quality_lightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = DualColorNetwork(in_channels=self.args.in_channels, gp=self.args.gp, hidden_channels=self.args.hidden_channels)
        self.quality_model = QualityNetwork(image_size=args.img_size, patch_size=args.patch_size, dim=args.dim, encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth, 
                                            heads=args.heads, channels=args.in_channels, drop_out=args.drop_out, emb_dropout=args.emb_dropout)
        self.select_quality_model = QualityNetwork(image_size=args.img_size, patch_size=args.patch_size, dim=args.dim, encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth,
                                                   heads=args.heads, channels=args.in_channels, drop_out=args.drop_out, emb_dropout=args.emb_dropout)
        assert self.args.dual_pretrain==True, "have to pretrain dual color network model"
        if self.args.dual_pretrain:
            self.model.load_state_dict(torch.load(args.dual_model_path))
        if self.args.pretrain:
            self.quality_model.load_state_dict(torch.load(args.model_path))
            self.select_quality_model.load_state_dict(torch.load(args.model_path))
        for param in self.select_quality_model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.select_quality_model.eval()
        self.weight = args.loss_weights
        self.gamma = args.ema_gamma
        self.init_loss_function()

    def init_loss_function(self):
        self.entropy = nn.CrossEntropyLoss()
        self.rank_entropy = nn.BCELoss()
        self.calculate_distance = nn.PairwiseDistance(p=2)

    def _loss_function(self, result, result_quality, low, low_quality, 
                       high, high_quality, rotate, labels):
        contrastive_loss = GroupContrastiveLoss(batch_size=result.shape[0],temperature=self.args.temperature).to(self.device)
        con_loss = contrastive_loss(low, high)
        target_distance = torch.ones(result.shape[0]).to(self.device)
        dis_high = self.calculate_distance(high, result)
        dis_low = self.calculate_distance(low, result)
        distance_loss = self.rank_entropy(torch.sigmoid(dis_high - dis_low), target_distance)
        target = torch.ones(size=(result.shape[0], 1)).to(self.device)
        diff_high = high_quality - result_quality
        diff_low = low_quality - result_quality
        rank_loss = self.rank_entropy(torch.sigmoid(diff_high - diff_low), target)
        rotate_loss = self.entropy(rotate, labels)
        return con_loss, rank_loss, rotate_loss, distance_loss

    def forward(self, input, input_low=None, input_high=None, noise_high=None, noise_low=None, rotation=False, infer=False, blur=True):
        self.model.eval()
        if infer:
            with torch.no_grad():
                ycbcr = self.model(input, only=True)
            _, result = self.quality_model(input, ycbcr, infer=True)
            return result
        else:
            batch_size = input.shape[0]
            high_input, high_ycbcr, low_input, low_ycbcr = calculate_quality_diff(self.model, self.select_quality_model, 
                                                                                input, input_low, input_high,  noise_high, noise_low, 
                                                                                blur, self.device)
            rotate_input, labels = rotate_batch(input, label='expand')
            with torch.no_grad():
                ycbcr = self.model(input, only=True)
                rotate_ycbcr = self.model(rotate_input, only=True)
            img = torch.cat([input, high_input, low_input, rotate_input], dim=0)
            ycbcr = torch.cat([ycbcr, high_ycbcr, low_ycbcr, rotate_ycbcr])
            result_quality, result, rotate_result = self.quality_model(img, ycbcr, batch_size=batch_size, infer=False)
            result_quality, high_quality, low_quality = result_quality.split(split_size=batch_size, dim=0)
            result, high_result, low_result = result.split(split_size=batch_size, dim=0)
            return result, result_quality, high_result, high_quality, low_result, low_quality, rotate_result, labels
    
    def training_step(self, batch, batch_idx):
        input, input_low, input_high, noise_high, noise_low = batch
        result, result_quality, high_result, high_quality, low_result, low_quality, rotate_result, labels = self(input, input_low, input_high, 
                                                                                                                noise_high, noise_low, rotation=True, blur=True)

        contrastive_loss, rank_loss, rotate_loss, distance_loss = self._loss_function(result, result_quality, 
                                                                       low_result, low_quality, 
                                                                       high_result, high_quality, 
                                                                       rotate_result, labels)
        
        if batch_idx%self.args.check_loss==0:
            print(f"contrastive_loss :{contrastive_loss:.4f} rank_loss : {rank_loss:.4f} rotate_loss : {rotate_loss:.4f}")
        if rotate_loss is not None:
            loss = self.weight[0]*contrastive_loss + self.weight[1]*rank_loss + self.weight[2]*rotate_loss + self.weight[3]*distance_loss
        else:
            loss = self.weight[0]*contrastive_loss + self.weight[1]*rank_loss + self.weight[3]*distance_loss
        return loss

    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if batch_idx != 0:
            self.update_ema()

    def on_validation_start(self):
        self.whole_contrastive = []
        self.whole_rank = []
        self.whole_rotate = []
        self.whole_dis = []

    def validation_step(self, batch, batch_idx):
        input, input_low, input_high, noise_high, noise_low = batch
        result, result_quality, high_result, high_quality, low_result, low_quality, rotate_result, labels = self(input, input_low, input_high, 
                                                                                                                noise_high, noise_low, rotation=True, blur=True)

        contrastive_loss, rank_loss, rotate_loss, distance_loss = self._loss_function(result, result_quality, 
                                                                       low_result, low_quality, 
                                                                       high_result, high_quality, 
                                                                       rotate_result, labels)
        self.whole_contrastive.append(torch.mean(contrastive_loss).detach().cpu())
        self.whole_rank.append(torch.mean(rank_loss).detach().cpu())
        self.whole_rotate.append(torch.mean(rotate_loss).detach().cpu())
        self.whole_dis.append(torch.mean(distance_loss).detach().cpu())

    def validation_epoch_end(self, outputs):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["loss_contrastive", np.mean(np.array(self.whole_contrastive))])
        table.add_row(["loss_rank", np.mean(np.array(self.whole_rank))])
        table.add_row(["loss_rotate", np.mean(np.array(self.whole_rotate))])
        table.add_row(["loss_dis", np.mean(np.array(self.whole_dis))])
        print(table)
        self.save()
    
    def on_test_start(self):
        self.whole_contrastive = []
        self.whole_rank = []
        self.whole_rotate = []
        self.whole_dis = []

    def test_step(self, batch, batch_idx):
        input, input_low, input_high, noise_high, noise_low = batch
        result, result_quality, high_result, high_quality, low_result, low_quality, rotate_result, labels = self(input, input_low, input_high, 
                                                                                                                noise_high, noise_low, rotation=True, blur=True)

        contrastive_loss, rank_loss, rotate_loss, distance_loss = self._loss_function(result, result_quality, 
                                                                       low_result, low_quality, 
                                                                       high_result, high_quality, 
                                                                       rotate_result, labels)
        self.whole_contrastive.append(torch.mean(contrastive_loss).detach().cpu())
        self.whole_rank.append(torch.mean(rank_loss).detach().cpu())
        self.whole_rotate.append(torch.mean(rotate_loss).detach().cpu())
        self.whole_dis.append(torch.mean(distance_loss).detach().cpu())
    
    def test_epoch_end(self, outputs):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["loss_contrastive", np.mean(np.array(self.whole_contrastive))])
        table.add_row(["loss_rank", np.mean(np.array(self.whole_rank))])
        table.add_row(["loss_rotate", np.mean(np.array(self.whole_rotate))])
        table.add_row(["loss_dis", np.mean(np.array(self.whole_dis))])
        print(table)
    
    def predict_step(self, batch, batch_idx):
        input = batch
        result = self(input, infer=True)
        return result
    
    def configure_optimizers(self):
        if self.args.optim=="SGD":
            optimizer = optim.SGD(self.quality_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = optim.AdamW(self.quality_model.parameters(), eps=self.args.eps, betas=self.args.betas,
                                            lr=self.args.lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler=="LambdaLR":
            scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:self.args.lambda_weight**epoch)
        else:
            scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.args.t_scheduler, T_mult=self.args.trigger_scheduler, 
                                                    eta_max=self.args.eta_scheduler, T_up=self.args.up_scheduler, gamma=self.args.gamma_scheduler)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def save(self):
        torch.save(self.quality_model.state_dict(), os.path.join(self.args.model_save_path, 'quality_model.pth'))
        torch.save(self.select_quality_model.state_dict(), os.path.join(self.args.model_save_path, 'select_quality_model.pth'))
        print(f'Model save epoch : {self.current_epoch}')

    def update_ema(self):
        for current, previous in zip(self.quality_model.parameters(), self.select_quality_model.parameters()):
            new_weight = self.gamma*previous + (1-self.gamma)*current
            previous.copy_(new_weight.detach())