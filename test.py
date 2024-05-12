import glob
import numpy as np
import torch
from PIL import Image
from common.logger import Logger, AverageMeter
from model.SSP_matching import SSP_MatchingNet

import torch.nn as nn
from data_util.datasets import FSSDataset
from common import utils
import argparse
from common.evaluation import Evaluator
from util.utils import Compute_iou


def parse_args():
    parser = argparse.ArgumentParser(description='Domain-Rectifying Adapter for CD-FSS')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal'],
                        help='training dataset')
    parser.add_argument('--backbone',
                        type=str,
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--nshot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--benchmark',
                        type=str,
                        default='fss',
                        choices=['fss','deepglobe','lung','isic'],
                        help='The benchmark to be tested')
    parser.add_argument('--test_datapath',
                        type=str,
                        default='./data/fss',
                        help='The path to the benchmark dataset')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='./outdir/models/pascal/best_model.pth',
                        help='Checkpoint path')

    args = parser.parse_args()
    return args


def test(model, dataloader, nshot):
    # Freeze randomness during testing for reproducibility if needed
    utils.fix_randseed(0)
    average_meter = AverageMeter(dataloader.dataset)

    for idx, batch in enumerate(dataloader):

        batch = utils.to_cuda(batch)

        sup_rgb = batch['support_imgs'][0]
        sup_msk = batch['support_masks'][0]
        qry_rgb = batch['query_img']
        qry_msk = batch['query_mask']


        pred = model([sup_rgb],[sup_msk] , qry_rgb, qry_msk, training=False)[0]
        pred_mask = torch.argmax(pred, dim=1)

        assert pred_mask.size() == batch['query_mask'].size()
        # Evaluate prediction
        area_inter, area_union = Evaluator.classify_prediction(pred_mask.clone(), batch)
        average_meter.update(area_inter, area_union, batch['class_id'], loss=None)
        average_meter.write_process(idx, len(dataloader), epoch=-1, write_batch_idx=1)

    # Write evaluation results
    average_meter.write_result('Test', 0)
    miou, fb_iou = average_meter.compute_iou()

    return miou, fb_iou

def main(args):
    Logger.initialize()
    FSSDataset.initialize(img_size=400, datapath=args.test_datapath)
    dataloader_test = FSSDataset.build_dataloader(args.benchmark, 1, 0, 'test', args.nshot)

    Logger.info(f'     ==> {len(dataloader_test)} testing samples')
    dim_ls = [128, 256, 512]
    model = SSP_MatchingNet(args.backbone,dim_ls)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint,strict=False)
    model.eval()
    Logger.log_params(model)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())
    model = nn.DataParallel(model)
    model.to(device)

    Evaluator.initialize()

    with torch.no_grad():
        test_miou, test_fb_iou = test(model, dataloader_test, args.nshot)
    Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))
    Logger.info('==================== Finished Testing ====================')



if __name__ == '__main__':

    args = parse_args()
    main(args)
