from utils.utils import seed_everything
import argparse
import os
import pandas as pd
from timm.models.layers import to_2tuple
from data.data_module import DataModule
import cv2
from core.function import DualColor_lightning
import pytorch_lightning as pl
import time

def main(args):
    train_csv = pd.read_csv(os.path.join(args.csv_path, 'train.csv'))
    valid_csv = pd.read_csv(os.path.join(args.csv_path, 'valid.csv'))
    test_csv = pd.read_csv(os.path.join(args.csv_path, 'test.csv')) if os.path.exists(os.path.join(args.csv_path, 'test.csv')) else valid_csv
    data_csv = {'train':train_csv, 'valid':valid_csv, 'test':test_csv}

    args.image_shape =  args.img_size if isinstance(args.img_size, tuple) else to_2tuple(args.img_size)

    data_module = DataModule(data_csv, img_size=args.image_shape, batch_size=args.batch_size, 
                             num_workers=args.workers, interpolation=cv2.INTER_LINEAR, stratch=True, scaleup=False)
    pl_model = DualColor_lightning(args=args)

    if args.types=="train":
        trainer = pl.Trainer(max_epochs=args.epoch, gpus=args.gpus, 
                             check_val_every_n_epoch=args.check_val, 
                             reload_dataloaders_every_n_epochs=1)
        start_time = time.time()
        trainer.fit(pl_model, data_module)
        print(f'training time : {time.time()-start_time}')
        start_time = time.time()
        trainer.test(pl_model, data_module)
        print(f'inference time : {time.time()-start_time}')
    elif args.types=="test":
        trainer = pl.Trainer(max_epochs=args.epoch, gpus=args.gpus)
        start_time = time.time()
        trainer.test(pl_model, data_module)
        print(f'inference time : {time.time()-start_time}')
    else:
        trainer = pl.Trainer(max_epochs=args.epoch, gpus=args.gpus)
        start_time = time.time()
        trainer.predict(pl_model, data_module)
        print(f'inference time : {time.time()-start_time}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        default=0, help='random seed')
    parser.add_argument('--csv_path', type=str, required=True, metavar="FILE", help='path to CSV file')
    parser.add_argument('--types', default='train', type=str, help='select types [train, test, predict]')
    parser.add_argument('--gpus', default=[1], type=int, nargs='+', help='GPU id to use.')
    parser.add_argument('--log-path', default='./log', type=str, help='Write Log Path')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of Batch Size')
    parser.add_argument('--epoch', default=300, type=int, help='Number of Epoch')
    parser.add_argument('--workers', type=int, default=1, help='Number of Workers')
    parser.add_argument('--check_val', type=int, default=10, help='Check validation step')
    parser.add_argument('--check_loss', type=int, default=100, help='Check training loss')

    parser.add_argument('--img_size', default=512, type=int, help='model input image size')
    parser.add_argument('--in_channels', default=3, type=int, help='model in channels')
    parser.add_argument('--gp', default=32, type=int, help='model global priors')
    parser.add_argument('--hidden_channels', default=64, type=int, help='model hidden channels')

    parser.add_argument('--loss_weights', default=[0.01, 1., 0.01, 0.1], nargs='+', type=int, help='[ycbcr, rgb, tv, color] weight')
    parser.add_argument('--optim', default='AdamW', type=str, help='type of optimizer')
    parser.add_argument('--momentum', default=0.95, type=float, help='SGD momentum')
    parser.add_argument('--lr', default=5e-5, type=float, help='Train Learning Rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='AdamW optimizer eps')
    parser.add_argument('--betas', default=(0.9, 0.999), help='AdamW optimizer betas')
    parser.add_argument('--weight_decay', default=0.95, type=float, help='AdamW optimizer weight decay')

    parser.add_argument('--scheduler', default='CosineWarmUp', type=str, help='type of Scheduler')
    parser.add_argument('--lambda_weight', default=0.975, type=float, help='LambdaLR Scheduler lambda weight')
    parser.add_argument('--t_scheduler', default=100, type=int, help='CosineAnnealingWarmUpRestarts optimizer time step')
    parser.add_argument('--trigger_scheduler', default=1, type=int, help='CosineAnnealingWarmUpRestarts optimizer T trigger')
    parser.add_argument('--eta_scheduler', default=5e-5, type=float, help='CosineAnnealingWarmUpRestarts optimizer eta max')
    parser.add_argument('--up_scheduler', default=10, type=int, help='CosineAnnealingWarmUpRestarts optimizer time Up')
    parser.add_argument('--gamma_scheduler', default=0.5, type=float, help='CosineAnnealingWarmUpRestarts optimizer gamma')

    parser.add_argument('--model_path', default='./weights/model.pth', type=str, help='Model Path')
    parser.add_argument('--model_save_path', default='./weights', type=str, help='Model Save Path')
    parser.add_argument('--img_save_path', default='./imgs', type=str, help='Result Img  Save Path')
    parser.add_argument('--pretrain', default=False, type=bool, help='Model Save Path')
    args = parser.parse_args()
    seed_everything(args.seed)
    main(args)