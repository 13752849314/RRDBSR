import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from options import parse, dict2str
from utils import setup_logger, makedir, tensor2img, save_img
from Archive import ArchiveSet
from loss import psnr, ssim

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options yaml file.')
args = parser.parse_args()
opt = parse(args.opt)

name = 'train_' + opt['name'] if opt['train'] else 'test_' + opt['name']
setup_logger(opt['name'], opt['logs_path'], name, level=logging.INFO,
             screen=True, tofile=True)

logger = logging.getLogger(opt['name'])
logger.info(dict2str(opt))

model_path = os.path.join(opt['model_path'], opt['model'])
device = torch.device('cuda') if opt['use_cuda'] and torch.cuda.is_available() else torch.device('cpu')

model = torch.load(model_path).to(device)
model.eval()

test_data = ArchiveSet(opt['data_path'], model='test', width=opt['size'][0], height=opt['size'][1],
                       grade=opt['grade'], scala=opt['scala'])
test = DataLoader(test_data, batch_size=1)

with torch.no_grad():
    val_path = os.path.join(opt['result_path'], f"{opt['model'].split('.')[0]}test")
    makedir([val_path])

    for item in test:
        x = item['LR'].to(device)
        y = item['HR'].to(device)
        out = model(x)
        sr_img = tensor2img(out[0])
        save_img(sr_img, os.path.join(val_path, f"{item['name'][0]}.png"))
        logger.info(f"image {item['name'][0]}\n"
                    f"psnr={psnr(out, y)}\n"
                    f"ssim={ssim(out, y)}")
