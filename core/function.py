import os
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
from models.network import DualColorNetwork
from core.criterion import total_variation_loss, color_loss
from core.optimizer import CosineAnnealingWarmUpRestarts
from prettytable import PrettyTable
from utils.utils import calculate_delta_e, calculate_psnr, calculate_ssim
import time

class DualColor_lightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = DualColorNetwork(in_channels=self.args.in_channels, gp=self.args.gp, hidden_channels=self.args.hidden_channels)
        if self.args.pretrain:
            self.model.load_state_dict(torch.load(args.model_path))
        self.distance_loss = nn.L1Loss()
        self.weight = args.loss_weights

    def _loss_function(self, ycbcr, rgb, target_ycbcr, target_rgb):
        loss_ycbcr = self.distance_loss(ycbcr, target_ycbcr)
        loss_rgb = self.distance_loss(rgb, target_rgb)
        loss_tv = total_variation_loss(rgb) 
        loss_color = color_loss(rgb, target_rgb)
        return loss_ycbcr, loss_rgb, loss_tv, loss_color
        
    def forward(self, input, infer):
        return self.model(input, infer=infer)
        
    def training_step(self, batch, batch_idx):
        input, target_ycbcr, target_rgb = batch
        ycbcr, rgb = self(input, infer=False)
        loss_ycbcr, loss_rgb, loss_tv, loss_color  = self._loss_function(ycbcr, rgb, target_ycbcr, target_rgb)
        # print(f"loss_ycbcr :{loss_ycbcr:.4f} loss_rgb : {loss_rgb:.4f} loss_tv : {loss_tv:.4f} loss_color : {loss_color:.4f}")
        return self.weight[0]*loss_ycbcr + self.weight[1]*loss_rgb + self.weight[2]*loss_tv + self.weight[3]*loss_color

    def on_validation_start(self):
        self.whole_ssim = []
        self.whole_psnr = []
        self.whole_delta = []
        self.whole_time = []

    def validation_step(self, batch, batch_idx):
        input, _, target_rgb = batch
        start_time = time.time()
        rgb = self(input, infer=True)
        self.whole_time.append(time.time()-start_time)
        delta_e = calculate_delta_e(rgb, target_rgb)
        psnr = calculate_psnr(rgb, target_rgb)
        ssim = calculate_ssim(rgb, target_rgb)
        self.whole_delta.append(torch.mean(delta_e))
        self.whole_psnr.append(torch.mean(psnr))
        self.whole_ssim.append(torch.mean(ssim))
        
    
    def validation_epoch_end(self, outputs):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["delta", np.mean(np.array(self.whole_delta))])
        table.add_row(["psnr", np.mean(np.array(self.whole_psnr))])
        table.add_row(["ssim", np.mean(np.array(self.whole_ssim))])
        table.add_row(["speed", np.mean(np.array(self.whole_time))])
        print(table)
        self.save()
    
    def on_test_start(self):
        self.whole_ycbcr = []
        self.whole_rgb = []

    def test_step(self, batch, batch_idx):
        input, _, target_rgb = batch
        rgb = self(input, infer=True)
        delta_e = calculate_delta_e(rgb, target_rgb)
        psnr = calculate_psnr(rgb, target_rgb)
        ssim = calculate_ssim(rgb, target_rgb)
        self.whole_delta.append(torch.mean(delta_e))
        self.whole_psnr.append(torch.mean(psnr))
        self.whole_ssim.append(torch.mean(ssim))
    
    def test_epoch_end(self, outputs):
        table = PrettyTable()
        table.field_names = ["Metric", "Value"]
        table.add_row(["delta", np.mean(np.array(self.whole_delta))])
        table.add_row(["psnr", np.mean(np.array(self.whole_psnr))])
        table.add_row(["ssim", np.mean(np.array(self.whole_ssim))])
        print(table)
        self.save()
    
    def predict_step(self, batch, batch_idx):
        input = batch
        rgb = self(input, infer=True)
        return rgb
    
    def configure_optimizers(self):
        if self.args.optim=="SGD":
            optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        else:
            optimizer = optim.AdamW(self.model.parameters(), eps=self.args.eps, betas=self.args.betas,
                                            lr=self.args.lr, weight_decay=self.args.weight_decay)
        # if self.args.scheduler=="LambdaLR":
        #     scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch:self.args.lambda_weight**epoch)
        # else:
        #     scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=self.args.t_scheduler, T_mult=self.args.trigger_scheduler, 
        #                                             eta_max=self.args.eta_scheduler, T_up=self.args.up_scheduler, gamma=self.args.gamma_scheduler)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.args.model_save_path, 'model.pth'))
        print(f'Model save epoch : {self.current_epoch}')
