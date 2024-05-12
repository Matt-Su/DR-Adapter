from model.SSP_matching import SSP_MatchingNet
from util.utils import count_params, set_seed, Compute_iou
import argparse
from copy import deepcopy
import os
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from tqdm import tqdm
from data_util.datasets import FSSDataset
from common.logger import Logger, AverageMeter
import torch.nn as nn


def parse_args():
    parser = argparse.ArgumentParser(description='Domain-Rectifying Adapter for CD-FSS')
    parser.add_argument('--batch-size',
                        type=int,
                        default=4,
                        help='batch size of training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='learning rate')
    parser.add_argument('--size',
                        type=int,
                        default=400,
                        help='Size of training samples')
    parser.add_argument('--backbone',
                        type=str,
                        choices=['resnet50'],
                        default='resnet50',
                        help='backbone of semantic segmentation model')
    parser.add_argument('--dataset',
                        type=str,
                        default='pascal',
                        choices=['pascal'],
                        help='training dataset')
    parser.add_argument('--train_datapath',
                        type=str,
                        default='./data/VOCdevkit',
                        help='The path of training dataset')

    parser.add_argument('--shot',
                        type=int,
                        default=1,
                        help='number of support pairs')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='random seed to generate tesing samples')
    parser.add_argument('--checkpoint_path',
                        type=str,
                        default='./outdir/models/Ori_SSP_trained_on_VOC.pth',
                        help='Checkpoint trained from original SSP that has not involved in adapter')

    parser.add_argument('--test_datapath',
                        type=str,
                        default='./data/chest',
                        help='The path of the benchmark')
    parser.add_argument('--benchmark',
                        type=str,
                        default='lung',
                        help = 'The benchmark to be tested')
    parser.add_argument('--global_noise_std',
                        type=float,
                        default=1.0)
    parser.add_argument('--local_noise_std',
                        type=float,
                        default=0.75)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print('\n' + str(args))

    save_path = 'outdir/models/%s' % (args.dataset)
    os.makedirs(save_path, exist_ok=True)

    FSSDataset.initialize(img_size=args.size, datapath=args.train_datapath)
    trainloader = FSSDataset.build_dataloader(args.dataset, args.batch_size, 4, 'trn', args.shot)
    FSSDataset.initialize(img_size=args.size, datapath=args.test_datapath)
    dataloader_val = FSSDataset.build_dataloader(args.benchmark, 15, 0, 'test', args.shot)
    dim_ls = [128, 256, 512]  ## The output channels of feature map for the first three stages

    model = SSP_MatchingNet(args.backbone,dim_ls,args.local_noise_std)

    x_param_loss=nn.L1Loss()
    criterion = CrossEntropyLoss(ignore_index=255)
    optimizer = SGD([param for param in model.parameters() if param.requires_grad],
                    lr=args.lr, momentum=0.9, weight_decay=5e-4)

    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint,strict=False)

    model = DataParallel(model).cuda()

    previous_best = 0
    Logger.initialize()

    datum_center_ls = []
    for epoch in range(5):

        print("\n==> Epoch %i, learning rate = %.5f\t\t\t\t Previous best = %.2f"
              % (epoch, optimizer.param_groups[0]["lr"], previous_best))

        global_noise_ls = []
        for dim in dim_ls:
            ones_mat = torch.ones((args.batch_size, dim, 1, 1))
            zeros_mat = torch.zeros((args.batch_size, dim, 1, 1))
            alpha = torch.normal(zeros_mat, args.global_noise_std * ones_mat).cuda()  # size: B, 1, C
            beta = torch.normal(zeros_mat, args.global_noise_std * ones_mat).cuda()  # size: B, 1, C
            pert = (alpha, beta)
            global_noise_ls.append(pert)
        model.train()

        total_loss = 0.0

        tbar = tqdm(trainloader)
        set_seed(args.seed)
        for idx, batch in enumerate(tbar):
            sup_rgb = batch['support_imgs'].squeeze().cuda()
            sup_msk = batch['support_masks'].long().squeeze().cuda()
            qry_rgb = batch['query_img'].cuda()
            qry_msk = batch['query_mask'].long().cuda()
            if epoch == 0:
                with torch.no_grad():  # Obtain the global statistical value for better Momentum update when epoch is 0
                    out_ls = model([sup_rgb], [sup_msk], qry_rgb, qry_msk, training=True, get_prot=True)
            else:
                out_ls = model([sup_rgb], [sup_msk], qry_rgb, qry_msk, training=True,global_noise_ls=global_noise_ls,
                                   datum_center_ls=datum_center_ls)
            mask_s = torch.cat([sup_msk], dim=0)

            x_style = out_ls[-1]
            for num in range(len(x_style)):
                current_center = x_style[num][0].mean(dim=0, keepdim=True)
                current_var = x_style[num][1].mean(dim=0, keepdim=True)
                if len(datum_center_ls)<len(dim_ls):
                    datum_center = current_center
                    datum_var = current_var
                    datum_center_ls.append([datum_center,datum_var])
                else:
                    datum_center_ls[num][0] = datum_center_ls[num][0] * 0.99 + 0.01 * current_center
                    datum_center_ls[num][1] = datum_center_ls[num][1] * 0.99 + 0.01 * current_var

            if epoch==0:
                continue

            x_param_ls = torch.zeros(1).cuda()
            if len(out_ls[-2])>0:
                for num in range(len(out_ls[-2])):
                    x_ori_mean, x_ori_var, x_new_mean, x_new_var,x_new_rect_mean,x_new_rect_var = out_ls[-2][num]
                    x_param_ls += (x_param_loss(x_ori_mean,x_new_mean) + x_param_loss(x_ori_var,x_new_var)+
                                  x_param_loss(x_ori_mean,x_new_rect_mean) + x_param_loss(x_ori_var,x_new_rect_var))
                x_param_ls = x_param_ls / len(out_ls[-2])

            loss = criterion(out_ls[0], qry_msk) + criterion(out_ls[1], qry_msk) + criterion(out_ls[2], mask_s) * 0.2
            loss += (2 * x_param_ls[0])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            tbar.set_description('Loss:%.2f' % (total_loss / (idx + 1)))
        # if epoch in [3,4]:
        #     optimizer.param_groups[0]['lr'] /= 10.0
        if epoch==0:
            continue

        model.eval()
        set_seed(args.seed)
        with torch.no_grad():
            test_miou, test_fb_iou = Compute_iou(model, dataloader_val, args.shot)
        Logger.info('mIoU: %5.2f \t FB-IoU: %5.2f' % (test_miou.item(), test_fb_iou.item()))


        if test_miou >= previous_best:
            best_model = deepcopy(model)
            previous_best = test_miou
            torch.save(best_model.module.state_dict(),os.path.join(save_path, 'best_model.pth'))

        torch.save(model.module.state_dict(),os.path.join(save_path, 'last_model.pth'))

if __name__ == '__main__':
    main()
