import model.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
import pdb
import random


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class SSP_MatchingNet(nn.Module):
    def __init__(self, backbone, dim_ls, local_noise_std=0.75):
        super(SSP_MatchingNet, self).__init__()
        backbone = resnet.__dict__[backbone](pretrained=True)

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = backbone.layer1, backbone.layer2, backbone.layer3
        self.layer_block = [self.layer0, self.layer1, self.layer2, self.layer3]

        self.local_noise_std = local_noise_std

        self.perturb_layers_idx = {}
        self.perturb_layers = [0, 1, 2]
        self.perturb_layer_dim = dim_ls

        self.DR_Adapter = nn.ModuleList()

        for idx, layer in enumerate(self.perturb_layers):
            self.perturb_layers_idx[layer] = idx
            Adapter = nn.Sequential(
                nn.Conv2d(self.perturb_layer_dim[idx], self.perturb_layer_dim[idx], kernel_size=7, stride=1,
                          dilation=2),
                LayerNorm2d(self.perturb_layer_dim[idx]),
                nn.ReLU(),
                nn.Conv2d(self.perturb_layer_dim[idx], self.perturb_layer_dim[idx], kernel_size=7, stride=1,
                          dilation=2),
                LayerNorm2d(self.perturb_layer_dim[idx]),
                nn.ReLU(),
                nn.Conv2d(self.perturb_layer_dim[idx], self.perturb_layer_dim[idx], kernel_size=7, stride=1,
                          dilation=2),
            )
            self.DR_Adapter.append(Adapter)

    def forward(self, img_s_list, mask_s_list, img_q, mask_q, training=True, global_noise_ls=[], datum_center_ls=[],
                get_prot=False):
        h, w = img_q.shape[-2:]
        if training:
            p_local = random.random()
            p_global = random.random()
        else:
            p_local = 1
            p_global = 1
        # feature maps of support images
        feature_s_list = []

        x_params = []
        s_0 = img_s_list[0]
        q_0 = img_q
        x_layer_style = []
        x_ori_vec = []
        for idx, layer in enumerate(self.layer_block):
            s_0 = layer(s_0)
            q_0 = layer(q_0)
            ## domain rectifying 
            if idx in self.perturb_layers:
                x_q_mean = q_0.mean(dim=(2, 3), keepdim=True).detach()
                x_s_mean = s_0.mean(dim=(2, 3), keepdim=True).detach()
                x_q_var = q_0.var(dim=(2, 3), keepdim=True).detach()
                x_s_var = s_0.var(dim=(2, 3), keepdim=True).detach()
                x_ori_mean = torch.cat((x_q_mean, x_s_mean), dim=0)
                x_ori_var = torch.cat((x_q_var, x_s_var), dim=0)
                x_layer_style.append([x_ori_mean, x_ori_var])   ## Obtain original style for global momentum update
                if get_prot:
                    continue

                x_q_disturb = q_0
                x_s_disturb = s_0
                flag = 0        ## flag == 1 : perturbation
                if training:
                    if p_local < 0.5:  # local perturb
                        x_q_disturb, x_s_disturb, alpha, beta = self.local_perturb(x_q_disturb,x_s_disturb)
                        flag = 1
                    if p_global < 0.5:  # global perturb
                        global_noise_alpha, global_noise_beta = global_noise_ls[self.perturb_layers_idx[idx]]
                        x_q_disturb, x_s_disturb, datum_center, datum_var = self.global_perturb(x_q_disturb, x_s_disturb,
                                                   global_noise_alpha, global_noise_beta, datum_center_ls, idx, x_layer_style)
                        flag = 1

                x_q_disturb_miu = x_q_disturb.mean(dim=(2, 3), keepdim=True)
                x_q_disturb_sigma = x_q_disturb.var(dim=(2, 3), keepdim=True)
                x_s_disturb_miu = x_s_disturb.mean(dim=(2, 3), keepdim=True)
                x_s_disturb_sigma = x_s_disturb.var(dim=(2, 3), keepdim=True)
                dist_statistics = (x_q_disturb_miu,x_q_disturb_sigma,x_s_disturb_miu, x_s_disturb_sigma)

                if (flag == 1) or (training == False):
                    x_q_rectify, x_s_rectify, x_q_rect_miu, x_q_rect_sigma, x_s_rect_miu, x_s_rect_sigma = self.domain_rectify(x_q_disturb, x_s_disturb,idx,dist_statistics)
                    q_0 = x_q_rectify
                    s_0 = x_s_rectify

                if training:
                    ## prepare for cyclic align loss
                    if flag == 1:
                        if p_local < 0.5:
                            second_rect_q_miu, second_rect_q_sigma = self.cyclic_rectify(idx, alpha, beta, x_q_rect_miu,
                                                                                         x_q_rect_sigma,q_0 )
                            second_rect_s_miu, second_rect_s_sigma = self.cyclic_rectify(idx, alpha, beta, x_s_rect_miu,
                                                                                         x_s_rect_sigma, s_0)
                        elif p_global < 0.5:
                            second_rect_q_miu, second_rect_q_sigma = self.cyclic_rectify(idx, global_noise_alpha,
                                                                                         global_noise_beta, x_q_rect_miu, x_q_rect_sigma, q_0)
                            second_rect_s_miu, second_rect_s_sigma = self.cyclic_rectify(idx, global_noise_alpha,
                                                                                         global_noise_beta, x_s_rect_miu, x_s_rect_sigma, s_0)
                            x_ori_mean = torch.cat((datum_center, datum_center), dim=0)
                            x_ori_var = torch.cat((datum_var, datum_var), dim=0)

                        second_rect_mean = torch.cat((second_rect_q_miu, second_rect_s_miu), dim=0)
                        second_rect_sigma = torch.cat((second_rect_q_sigma, second_rect_s_sigma),
                                                   dim=0)

                        first_rect_mean = torch.cat((x_q_rect_miu, x_s_rect_miu), dim=0)
                        first_rect_sigma = torch.cat((x_q_rect_sigma, x_s_rect_sigma), dim=0)

                        x_param = (x_ori_mean, x_ori_var, first_rect_mean, first_rect_sigma, second_rect_mean, second_rect_sigma)
                        x_params.append(x_param)

        feature_q = q_0
        feature_s_list.append(s_0)

        # foreground(target class) and background prototypes pooled from K support features
        feature_fg_list = []
        feature_bg_list = []
        supp_out_ls = []
        for k in range(len(img_s_list)):
            feature_fg = self.masked_average_pooling(feature_s_list[k],
                                                     (mask_s_list[k] == 1).float())[None, :]
            feature_bg = self.masked_average_pooling(feature_s_list[k],
                                                     (mask_s_list[k] == 0).float())[None, :]
            feature_fg_list.append(feature_fg)
            feature_bg_list.append(feature_bg)

            if self.training:
                supp_similarity_fg = F.cosine_similarity(feature_s_list[k], feature_fg.squeeze(0)[..., None, None],
                                                         dim=1)
                supp_similarity_bg = F.cosine_similarity(feature_s_list[k], feature_bg.squeeze(0)[..., None, None],
                                                         dim=1)
                supp_out = torch.cat((supp_similarity_bg[:, None, ...], supp_similarity_fg[:, None, ...]), dim=1) * 10.0

                supp_out = F.interpolate(supp_out, size=(h, w), mode="bilinear", align_corners=True)
                supp_out_ls.append(supp_out)

        # average K foreground prototypes and K background prototypes
        FP = torch.mean(torch.cat(feature_fg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)
        BP = torch.mean(torch.cat(feature_bg_list, dim=0), dim=0).unsqueeze(-1).unsqueeze(-1)

        # measure the similarity of query features to fg/bg prototypes
        out_0 = self.similarity_func(feature_q, FP, BP)

        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.SSP_func(feature_q, out_0)

        FP_1 = FP * 0.5 + SSFP_1 * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        out_1 = self.similarity_func(feature_q, FP_1, BP_1)

        out_1 = F.interpolate(out_1, size=(h, w), mode="bilinear", align_corners=True)

        out_ls = [out_1]

        if self.training:
            fg_q = self.masked_average_pooling(feature_q, (mask_q == 1).float())[None, :].squeeze(0)
            bg_q = self.masked_average_pooling(feature_q, (mask_q == 0).float())[None, :].squeeze(0)

            self_similarity_fg = F.cosine_similarity(feature_q, fg_q[..., None, None], dim=1)
            self_similarity_bg = F.cosine_similarity(feature_q, bg_q[..., None, None], dim=1)
            self_out = torch.cat((self_similarity_bg[:, None, ...], self_similarity_fg[:, None, ...]), dim=1) * 10.0

            self_out = F.interpolate(self_out, size=(h, w), mode="bilinear", align_corners=True)
            supp_out = torch.cat(supp_out_ls, 0)
            out_ls.append(self_out)
            out_ls.append(supp_out)

        out_ls.append(x_ori_vec)
        out_ls.append(x_params)
        out_ls.append(x_layer_style)

        return out_ls

    def cyclic_rectify(self,idx, alpha, beta, x_rect_miu, x_rect_sigma, feature):

        second_dist_mean = (1 + beta) * x_rect_miu
        second_dist_sigma = (1 + alpha) * x_rect_sigma

        second_dist = ((feature - x_rect_miu) / (
                1e-6 + x_rect_sigma)) * second_dist_sigma + second_dist_mean

        second_rect = self.DR_Adapter[self.perturb_layers_idx[idx]](second_dist.detach())
        second_rect_beta = second_rect.mean(dim=(2, 3), keepdim=True)
        second_rect_alpha = second_rect.var(dim=(2, 3), keepdim=True)

        second_rect_miu = (1 + second_rect_beta) * second_dist_mean
        second_rect_sigma = (1 + second_rect_alpha) * second_dist_sigma

        return second_rect_miu, second_rect_sigma   #get cyclic rectifying statistics


    def local_perturb(self, x_q_disturb, x_s_disturb):
        zeros_mat = torch.zeros(x_q_disturb.mean(dim=(2, 3), keepdim=True).shape)
        ones_mat = torch.ones(x_q_disturb.mean(dim=(2, 3), keepdim=True).shape)
        alpha = torch.normal(zeros_mat, self.local_noise_std * ones_mat).cuda()  # size: B, 1, 1, C
        beta = torch.normal(zeros_mat, self.local_noise_std * ones_mat).cuda()  # size: B, 1, 1, C

        local_x_q_disturb = ((1 + alpha) * x_q_disturb - alpha * x_q_disturb.mean(dim=(2, 3),
                            keepdim=True) + beta * x_q_disturb.mean(dim=(2, 3), keepdim=True))

        local_x_s_disturb = ((1 + alpha) * x_s_disturb - alpha * x_s_disturb.mean(dim=(2, 3),
                            keepdim=True) + beta * x_s_disturb.mean(dim=(2, 3), keepdim=True))

        return local_x_q_disturb, local_x_s_disturb, alpha, beta

    def global_perturb(self, x_q_disturb, x_s_disturb, global_noise_alpha,global_noise_beta, datum_center_ls,idx, x_layer_style):
        if len(datum_center_ls) > 0:
            datum_center, datum_var = datum_center_ls[self.perturb_layers_idx[idx]]
        else:
            datum_center, datum_var = x_layer_style[self.perturb_layers_idx[idx]]
            datum_center = datum_center.mean(dim=0, keepdim=True)
            datum_var = datum_var.mean(dim=0, keepdim=True)

        datum_center = datum_center.repeat(len(x_q_disturb), 1, 1, 1).detach()
        datum_var = datum_var.repeat(len(x_s_disturb), 1, 1, 1).detach()

        global_x_s_disturb = (1 + global_noise_alpha) * x_s_disturb - global_noise_alpha * datum_center + global_noise_beta * datum_center
        global_x_q_disturb = (1 + global_noise_alpha) * x_q_disturb - global_noise_alpha * datum_center + global_noise_beta * datum_center

        return global_x_q_disturb, global_x_s_disturb, datum_center, datum_var

    def domain_rectify(self, x_q_disturb, x_s_disturb, idx, dist_statistics):
        x_q_disturb_miu, x_q_disturb_sigma, x_s_disturb_miu, x_s_disturb_sigma = dist_statistics

        x_q_rect = self.DR_Adapter[self.perturb_layers_idx[idx]](x_q_disturb.detach())

        x_q_rect_beta = x_q_rect.mean(dim=(2, 3), keepdim=True)  ##[bs*2,1,768]
        x_q_rect_alpha = x_q_rect.var(dim=(2, 3), keepdim=True)

        x_q_rect_miu = (1 + x_q_rect_beta) * x_q_disturb_miu
        x_q_rect_sigma = (1 + x_q_rect_alpha) * x_q_disturb_sigma
        x_q_rectify = ((x_q_disturb - x_q_disturb_miu) / (
                1e-6 + x_q_disturb_sigma)) * x_q_rect_sigma + x_q_rect_miu

        x_s_rect = self.DR_Adapter[self.perturb_layers_idx[idx]](x_s_disturb.detach())

        x_s_rect_beta = x_s_rect.mean(dim=(2, 3), keepdim=True)
        x_s_rect_alpha = x_s_rect.var(dim=(2, 3), keepdim=True)

        x_s_rect_miu = ((1 + x_s_rect_beta)) * x_s_disturb_miu
        x_s_rect_sigma = ((1 + x_s_rect_alpha)) * x_s_disturb_sigma
        x_s_rectify = ((x_s_disturb - x_s_disturb_miu) / (
                1e-6 + x_s_disturb_sigma)) * x_s_rect_sigma + x_s_rect_miu

        return x_q_rectify, x_s_rectify, x_q_rect_miu, x_q_rect_sigma, x_s_rect_miu, x_s_rect_sigma

    def SSP_func(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7  # 0.9 #0.6
            bg_thres = 0.6  # 0.6
            cur_feat = feature_q[epi].view(1024, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi] > fg_thres)]
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices]
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi] > bg_thres)]
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices]
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True)
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True)
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True)

            cur_feat_norm_t = cur_feat_norm.t()  # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t())
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t())

            fg_proto_local = fg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)
            bg_proto_local = bg_proto_local.t().view(1024, f_h, f_w).unsqueeze(0)

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature
