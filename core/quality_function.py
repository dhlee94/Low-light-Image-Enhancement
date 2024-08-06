import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from models.network import DualColorNetwork
from models.quality_network import QualityNetwork
from core.criterion import GroupContrastiveLoss, calculate_rank_loss
from core.optimizer import CosineAnnealingWarmUpRestarts
from utils.utils import rotate_batch
from prettytable import PrettyTable
import torchvision.transforms as T
import time

class Quality_lightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = DualColorNetwork(in_channels=self.args.in_channels, gp=self.args.gp, hidden_channels=self.args.hidden_channels)
        self.quality_model = QualityNetwork(encoder=args.encoder, num_channels=args.num_channels, hidden_div=args.hidden_div)
        self.select_quality_model = QualityNetwork(encoder=args.encoder, num_channels=args.num_channels, hidden_div=args.hidden_div)
        self.select_quality_model.eval()
        assert self.args.dual_pretrain==True, "have to pretrain dual color network model"
        if self.args.dual_pretrain:
            self.model.load_state_dict(torch.load(args.dual_model_path))
        self.model.eval()
        self.contrastive_loss = GroupContrastiveLoss(temperature=args.temperature)
        self.entropy = nn.CrossEntropyLoss()
        self.weight = args.loss_weights
        self.gamma = args.ema_gamma
    def _loss_function(self, result, low, high, rotate, labels):
        contrastive_loss = self.contrastive_loss(low, high)
        rank_loss = calculate_rank_loss(result, high, low)
        if rotate is not None and labels is not None:
            rotate_loss = self.entropy(rotate, labels)
            return contrastive_loss, rank_loss, rotate_loss
        else:
            return contrastive_loss, rank_loss, None

    def forward(self, input, input_low=None, input_high=None, noise_high=None, noise_low=None, rotation=False, infer=False, blur=True):
        high = []
        low = []
        with torch.no_grad():
            ycbcr = self.model(input, only=True)
        result = self.quality_model(input.clone(), ycbcr)
        if infer:
            return result
        device = input.get_device()
        B, C, H, W = input.shape
        data_device = torch.device(f"cuda:{device}") if device >= 0 else torch.device("cpu")
        sigma1 = 40 + np.random.random() * 20
        sigma2 = 5 + np.random.random() * 15
        with torch.no_grad():
            self.select_quality_model.eval()
            if blur:
                blur_high = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma1))(input).to(data_device)
                blur_low = T.GaussianBlur(kernel_size=(5, 5), sigma=(sigma2))(input).to(data_device)
                high.append(blur_high)
                low.append(blur_low)
            if input_low is not None and input_high is not None:
                high.append(input_high)
                low.append(input_low)
            if noise_high is not None and noise_low is not None:
                high.append(noise_high)
                low.append(noise_low)
            length = len(high)
            high = torch.cat(high, dim=0)
            low  = torch.cat(low, dim=0)
            high_ycbcr = self.model(high, only=True)
            low_ycbcr = self.model(low, only=True)
            quality_high = self.quality_model(high, high_ycbcr.detach())
            quality_low = self.quality_model(low, low_ycbcr.detach())
            high = high.reshape(-1, B, C, H, W).transpose(1, 0)
            low = low.reshape(-1, B, C, H, W).transpose(1, 0)
            high_ycbcr = high_ycbcr.reshape(-1, B, C, H, W).transpose(1, 0)
            low_ycbcr = low_ycbcr.reshape(-1, B, C, H, W).transpose(1, 0)
            diff = torch.abs(quality_high - quality_low).reshape(length, B, -1).sum(dim=-1).permute(1, 0)
            index = [torch.argmax(data).item() for data in diff]
            high_input = torch.stack([high[idx, index[idx], ...] for idx in range(B)] , dim=0).contiguous()
            low_input = torch.stack([low[idx, index[idx], ...] for idx in range(B)], dim=0).contiguous()
            high_ycbcr = torch.stack([high_ycbcr[idx, index[idx], ...] for idx in range(B)] , dim=0).contiguous()
            low_ycbcr = torch.stack([low_ycbcr[idx, index[idx], ...] for idx in range(B)], dim=0).contiguous()
        high_result = self.quality_model(high_input, high_ycbcr.detach())
        low_result = self.quality_model(low_input, low_ycbcr.detach())
        
        if rotation:
            rotate_input, labels = rotate_batch(input, label='expand')
            rotate_ycbcr = self.model(rotate_input, only=True)
            rotate_result = self.quality_model(rotate_input, rotate_ycbcr, rotation=True)
            return result, high_result, low_result, rotate_result, labels
        return result, high_result, low_result, None, None
    
    def training_step(self, batch, batch_idx):
        input, input_low, input_high, noise_high, noise_low = batch
        result, high_result, low_result, rotate_result, labels = self(input, input_low, input_high, noise_high, noise_low, rotation=True, blur=True)

        contrastive_loss, rank_loss, rotate_loss = self._loss_function(result, low_result, high_result, rotate_result, labels)

        if batch_idx%self.args.check_loss==0:
            print(f"contrastive_loss :{contrastive_loss:.4f} rank_loss : {rank_loss:.4f} rotate_loss : {rotate_loss:.4f}")
        if rotate_loss is None:
            return self.weight[0]*contrastive_loss + self.weight[1]*rank_loss + self.weight[2]*rotate_loss
        else:
            return self.weight[0]*contrastive_loss + self.weight[1]*rank_loss
    
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        if batch_idx != 0:
            self.update_ema()

    def on_validation_start(self):
        self.whole_contrastive = []
        self.whole_rank = []
        self.whole_rotate = []

    def validation_step(self, batch, batch_idx):
        input, input_low, input_high, noise_high, noise_low = batch
        result, high_result, low_result, rotate_result, labels = self(input, input_low, input_high, noise_high, noise_low, rotation=True, blur=True)

        contrastive_loss, rank_loss, rotate_loss = self._loss_function(result, low_result, high_result, rotate_result, labels)

        self.whole_contrastive.append(torch.mean(contrastive_loss).detach().cpu())
        self.whole_rank.append(torch.mean(rank_loss).detach().cpu())
        self.whole_rotate.append(torch.mean(rotate_loss).detach().cpu())
    
    def validation_epoch_end(self, outputs):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["loss_contrastive", np.mean(np.array(self.whole_contrastive))])
        table.add_row(["loss_rank", np.mean(np.array(self.whole_rank))])
        table.add_row(["loss_rotate", np.mean(np.array(self.whole_rotate))])
        print(table)
        self.save()
    
    def on_test_start(self):
        self.whole_contrastive = []
        self.whole_rank = []
        self.whole_rotate = []

    def test_step(self, batch, batch_idx):
        input, input_low, input_high, noise_high, noise_low = batch
        result, high_result, low_result, rotate_result, labels = self(input, input_low, input_high, noise_high, noise_low, rotation=True, blur=True)

        contrastive_loss, rank_loss, rotate_loss = self._loss_function(result, low_result, high_result, rotate_result, labels)

        self.whole_contrastive.append(torch.mean(contrastive_loss).detach().cpu())
        self.whole_rank.append(torch.mean(rank_loss).detach().cpu())
        self.whole_rotate.append(torch.mean(rotate_loss).detach().cpu())
    
    def test_epoch_end(self, outputs):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["loss_contrastive", np.mean(np.array(self.whole_contrastive))])
        table.add_row(["loss_rank", np.mean(np.array(self.whole_rank))])
        table.add_row(["loss_rotate", np.mean(np.array(self.whole_rotate))])
        print(table)
    
    def predict_step(self, batch, batch_idx):
        input = batch
        result = self(input, infer=True)
        return result
    
    def configure_optimizers(self):
        if self.args.optim=="SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = optim.AdamW(self.model.parameters(), eps=self.args.eps, betas=self.args.betas,
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
        with torch.no_grad():
            for current, previous in zip(self.quality_model.parameters(), self.select_quality_model.parameters()):
                new_weight = self.gamma*previous.detach() + (1-self.gamma)*current.detach()
                previous.copy_(new_weight)