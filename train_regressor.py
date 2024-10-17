import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import Ridge
import pickle
from models.network import DualColorNetwork
from models.quality_network import QualityNetwork
from utils.utils import seed_everything
from PIL import Image
import utils.augments as A
from tqdm import tqdm
import torch
import cv2
import os

def convert_to_serializable(data):
    if isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(v) for v in data]
    elif isinstance(data, np.float32):  # float32 타입을 float으로 변환
        return float(data)
    else:
        return data
    
def main(args):
    device = torch.device(f"cuda:{args.gpus}")
    model = DualColorNetwork(in_channels=args.in_channels, gp=args.gp, hidden_channels=args.hidden_channels)
    quality_model = QualityNetwork(image_size=args.img_size, patch_size=args.patch_size, dim=args.dim, encoder_depth=args.encoder_depth, decoder_depth=args.decoder_depth, 
                                            heads=args.heads, channels=args.in_channels, drop_out=args.drop_out, emb_dropout=args.emb_dropout)
    
    model.load_state_dict(torch.load(args.dual_model_path))
    quality_model.load_state_dict(torch.load(args.quality_model))
    model.to(device)
    quality_model.to(device)
    model.eval()
    quality_model.eval()
    path = "/home/dhlee/image-enhancement/kadid/image_labeled_by_per_noise.csv"
    image_path = '/home/dhlee/image-enhancement/kadid/images'
    image_name = pd.read_csv(path)['image'].values
    image_value = pd.read_csv(path)['dmos'].values

    transform = A.Compose([
        A.Resize(size=(args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR, scaleup=True, stratch=True),
        A.Normalize()])
    
    feat = []
    regression = []
    with torch.no_grad():
        for name in tqdm(image_name):
            img = Image.open(os.path.join(image_path, name)).convert("RGB")
            input = transform(np.array(img)).unsqueeze(0).to(device)
            ycbcr = model(input, only=True)
            regression_result, result = quality_model(input, ycbcr, infer=True)
            regression.append(regression_result)
            feat.append(result)

    feat = torch.cat(feat).detach().cpu().numpy()
    regression = torch.cat(regression).detach().cpu().numpy()
    #train regression
    reg = Ridge(alpha=args.alpha).fit(feat, image_value)
    pickle.dump(reg, open(args.model_save_path,'wb'))
    predict = pickle.load(open(args.model_save_path,'rb'))
    predictions = predict.predict(feat)
    dictionary = {"regression": list(regression), "predict": list(predictions), "label": image_value}
    dictionary = convert_to_serializable(dictionary)
    df = pd.DataFrame.from_dict(dictionary)
    df.to_csv("./regression_result.csv")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        default=0, help='random seed')
    parser.add_argument('--types', default='train', type=str, help='select types [train, test, predict]')
    parser.add_argument('--log-path', default='./log', type=str, help='Write Log Path')
    parser.add_argument('--gpus', default=0, type=int, nargs='+', help='GPU id to use.')    
    parser.add_argument('--batch_size', type=int, default=8, help='Number of Batch Size')
    parser.add_argument('--epoch', default=300, type=int, help='Number of Epoch')
    parser.add_argument('--workers', type=int, default=1, help='Number of Workers')
    parser.add_argument('--check_val', type=int, default=10, help='Check validation step')
    parser.add_argument('--check_loss', type=int, default=100, help='Check training loss')
    parser.add_argument('--temperature', type=int, default=1, help='the temperature in Group contrastive loss')
    parser.add_argument('--patch_size', type=int, default=8, help='VIT patch size')
    parser.add_argument('--dim', type=int, default=512, help='VIT patch embedding dim')
    parser.add_argument('--encoder_depth', type=int, default=2, help='VIT layer depth, more than 1')
    parser.add_argument('--decoder_depth', type=int, default=1, help='VIT layer depth, more than 1')
    parser.add_argument('--heads', type=int, default=8, help='VIT num head')
    parser.add_argument('--drop_out', type=float, default=0., help='VIT drop out percentage')
    parser.add_argument('--emb_dropout', type=float, default=0., help='VIT patch embedding dim drop out percentage')

    parser.add_argument('--img_size', default=256, type=int, help='model input image size')
    parser.add_argument('--in_channels', default=3, type=int, help='model in channels')
    parser.add_argument('--gp', default=32, type=int, help='model global priors')
    parser.add_argument('--hidden_channels', default=64, type=int, help='model hidden channels')

    parser.add_argument('--alpha', default=1.0, type=float, help='regression model parameter that controls the strength of regularization')

    parser.add_argument('--loss_weights', default=[1., 1e-2, 1e-2], nargs='+', type=int, help='[contrastive, rank, rotation] weight')
    parser.add_argument('--optim', default='AdamW', type=str, help='type of optimizer')
    parser.add_argument('--momentum', default=0.95, type=float, help='SGD momentum')
    parser.add_argument('--lr', default=1.25e-4, type=float, help='Train Learning Rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='AdamW optimizer eps')
    parser.add_argument('--betas', default=(0.9, 0.999), help='AdamW optimizer betas')
    parser.add_argument('--weight_decay', default=0.95, type=float, help='AdamW optimizer weight decay')
    parser.add_argument('--ema_gamma', type=float, default=0.9, help='check diff model ema update gamma value')

    parser.add_argument('--scheduler', default='LambdaLR', type=str, help='type of Scheduler')
    parser.add_argument('--lambda_weight', default=0.975, type=float, help='LambdaLR Scheduler lambda weight')
    parser.add_argument('--t_scheduler', default=100, type=int, help='CosineAnnealingWarmUpRestarts optimizer time step')
    parser.add_argument('--trigger_scheduler', default=1, type=int, help='CosineAnnealingWarmUpRestarts optimizer T trigger')
    parser.add_argument('--eta_scheduler', default=1.25e-4, type=float, help='CosineAnnealingWarmUpRestarts optimizer eta max')
    parser.add_argument('--up_scheduler', default=10, type=int, help='CosineAnnealingWarmUpRestarts optimizer time Up')
    parser.add_argument('--gamma_scheduler', default=0.5, type=float, help='CosineAnnealingWarmUpRestarts optimizer gamma')

    parser.add_argument('--dual_model_path', default='./weights/model.pth', type=str, help='Model Path')
    parser.add_argument('--quality_model', default='./weights/quality_model.pth', type=str, help='Model Path')
    parser.add_argument('--model_save_path', default='./weights/regression.save', type=str, help='Model Save Path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    main(args)