import argparse
import logging
import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from options import parse, dict2str
from utils import setup_logger, makedir, tensor2img, save_img

from Archive import ArchiveSet
from RRDBNet import RRDBNet
from loss import psnr, ssim
from TLConv import TLConv

parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options yaml file.')
args = parser.parse_args()
opt = parse(args.opt)

name = 'train_' + opt['name'] if opt['train'] else 'test_' + opt['name']
setup_logger(opt['name'], opt['logs_path'], name, level=logging.INFO,
             screen=True, tofile=True)

logger = logging.getLogger(opt['name'])
logger.info(dict2str(opt))

epochs = opt['epochs']
in_ch = opt['in_ch']
out_ch = opt['out_ch']
nb = opt['nb']
nf = opt['nf']
gc = opt['gc']
sf = opt['sf']
grade = opt['grade']
padding_mode = opt['padding_mode']
bias = opt['bias']

train_data = ArchiveSet(opt['data_path'], model='train', width=opt['size'][0], height=opt['size'][1],
                        grade=grade, scala=opt['scala'])
train = DataLoader(train_data, batch_size=opt['batch_size'])

val_data = ArchiveSet(opt['data_path'], model='val', width=opt['size'][0], height=opt['size'][1],
                      grade=grade, scala=opt['scala'])
val = DataLoader(val_data, batch_size=opt['batch_size'])

logger.info(f"data Set {train_data.__class__.__name__}\n"
            f"model is {train_data.model}\n"
            f"have images {len(train_data)}")
logger.info(f"data Set {val_data.__class__.__name__}\n"
            f"model is {val_data.model}\n"
            f"have images {len(val_data)}")

if opt['conv'] == 'Conv2d':
    conv = nn.Conv2d
elif opt['conv'] == 'TLConv':
    conv = TLConv
else:
    raise ValueError(f"{opt['conv']} Conv is not support!")
device = torch.device('cuda') if opt['use_cuda'] and torch.cuda.is_available() else torch.device('cpu')

model = RRDBNet(in_ch, out_ch, nf=nf, nb=nb, gc=gc, conv=conv, padding_mode=padding_mode, bias=bias)
if opt['gpus'] > 1:
    model = nn.DataParallel(model).to(device)
else:
    model = model.to(device)
criteon = nn.MSELoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=opt['lr'])
logger.info(model)

logger.info('start training!')

for epoch in range(epochs):
    model.train()
    total_loss = []

    for batchidx, item in enumerate(train):
        x, label = item['LR'].to(device), item['HR'].to(device)
        out = model(x)
        loss = criteon(out, label)
        total_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        logger.info(f"epoch={epoch + 1}  batch={batchidx + 1}  loss={loss.item()}")

    logger.info(f"epoch={epoch + 1}  avg_loss={sum(total_loss) / len(total_loss)}")
    if (epoch + 1) % sf == 0:
        torch.save(model, os.path.join(opt['model_path'], f"{epoch + 1}_{epochs}.pth"))

    with torch.no_grad():
        model.eval()
        val_path = os.path.join(opt['result_path'], f"{epoch + 1}")
        makedir([val_path])
        psnr_list = []
        ssim_list = []
        loss_list = []
        for item in val:
            x = item['LR'].to(device)
            y = item['HR'].to(device)
            out = model(x)
            psnr_list.append(psnr(out, y))
            ssim_list.append(ssim(out, y))
            loss_list.append(criteon(out, y).item())
            sr_img = tensor2img(out[0])
            save_img(sr_img, os.path.join(val_path, f"{item['name'][0]}.png"))
        logger.info(f'avg_psnr={sum(psnr_list) / len(psnr_list)}'
                    f'avg_ssim={sum(ssim_list) / len(ssim_list)}'
                    f'avg_loss={sum(loss_list) / len(loss_list)}')
