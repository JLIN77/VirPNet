import torch
import numpy as np
import torch.nn as nn


min_shape = (1224, 370)
im_area = min_shape[0] * min_shape[1]

# discuss: should foreground point and background point re-weight?


# def loss(f_map, b_map, gt_map, pred_map, gamma=1, average=True):
#     # f_np = f_map.detach().cpu().numpy()[0, :, :]
#     # b_np = b_map.detach().cpu().numpy()[0, :, :]
#     # gt_np = gt_map.detach().cpu().numpy()[0, :, :]
#     # pred_f_np = pred_map[0].detach().cpu().numpy()[0, 0, :, :]
#     # pred_b_np = pred_map[0].detach().cpu().numpy()[0, 1, :, :]
#     # pred_dl_np = pred_map[1].detach().cpu().numpy()[0, 0, :, :]
#     # print(pred_dl_np.shape)
#     # np.savez('gt_pred.npz', f_map=f_np, b_map=b_np, gt_map=gt_np, pred_f_map=pred_f_np, pred_b_map=pred_b_np, pred_dl_map=pred_dl_np)
#     # exit()
#
#     cls_map, reg_map = pred_map[0], pred_map[1]
#
#     # cls loss
#     # print(f_map.shape, b_map.shape, cls_map.shape)  # torch.Size([2, 1224, 370]) torch.Size([2, 1224, 370]) torch.Size([2, 2, 1224, 370])
#     loss1 = nn.BCELoss()
#     pred_cls_f_map, pred_cls_b_map = cls_map[:, 0, :, :] * f_map, cls_map[:, 1, :, :] * b_map
#     cls_f_loss = loss1(pred_cls_f_map.float(), f_map.float())
#     cls_b_loss = loss1(pred_cls_b_map.float(), b_map.float())
#     # print(cls_f_loss, cls_b_loss)
#     cls_loss = 10000 * (cls_f_loss + cls_b_loss)
#
#     # reg loss
#     # print(gt_map.shape, reg_map.shape)   # torch.Size([2, 1224, 370]) torch.Size([2, 1, 1224, 370])
#     reg_map = reg_map.squeeze(axis=1)
#     # reg_f_loss = torch.sum(torch.pow(gt_map-reg_map, 2) * f_map)
#     # reg_b_loss = torch.sum(torch.pow(gt_map-reg_map, 2) * b_map)
#     # print(reg_f_loss, reg_b_loss)
#     err = 1
#     gt_map_bin = torch.sum(torch.where(gt_map != 0, 1, 0))
#     # reg_loss = (reg_f_loss + gamma * reg_b_loss) / (gt_map_bin + err)
#
#     reg_loss = torch.sum(torch.pow(gt_map - reg_map, 2)) / (gt_map_bin + err)
#
#     loss_all = cls_loss + reg_loss
#     # print('cls_loss, reg_loss : {}, {}'.format(round(cls_loss.item(), 6), round(reg_loss.item(), 6)))
#
#     if average:
#         N = gt_map.shape[0]
#         return loss_all / N
#
#     return loss_all, cls_loss, reg_loss

import pandas as pd


# def loss(gt_map, pred_map, uv_map, gamma=0.2, average=True):
#
#     pred_cls_map, pred_reg_map, pred_ref_map = pred_map[:, 0:2, :, :], pred_map[:, 2, :, :], pred_map[:, 3, :, :]
#     gt_cls_map, gt_reg_map, gt_ref_map = gt_map[:, 0:2, :, :], gt_map[:, 2, :, :], gt_map[:, 3, :, :]
#
#     gt_map_bin = torch.sum(uv_map)
#
#     ################################################################################
#     # cls loss
#     loss1 = nn.BCELoss()
#     pred_cls_f_map = pred_cls_map[:, 0, :, :]
#     pred_cls_b_map = pred_cls_map[:, 1, :, :]
#     gt_cls_f_map = gt_cls_map[:, 0, :, :]
#     gt_cls_b_map = gt_cls_map[:, 1, :, :]
#
#     cls_f_loss = loss1(pred_cls_f_map, gt_cls_f_map)
#     cls_b_loss = loss1(pred_cls_b_map, gt_cls_b_map)
#     # print(cls_f_loss, cls_b_loss)
#     cls_loss = (cls_f_loss + cls_b_loss) * im_area / gt_map_bin
#
#     ###################################################################################
#     # reg loss
#     err = 0.0
#     # # print(gt_map_bin)
#     reg_loss = torch.sum(torch.pow(gt_reg_map - pred_reg_map, 2)) / (gt_map_bin + err)
#     # loss2 = nn.L1Loss()
#     # reg_loss = 10000 * loss2(pred_reg_map, gt_reg_map)
#
#     ###################################################################################
#     # # ref loss
#     # err = 0.0
#     # # # print(gt_map_bin)
#     # ref_loss = torch.sum(torch.pow(gt_ref_map - pred_ref_map, 2)) / (gt_map_bin + err)
#
#     ##################################################################################
#     loss_all = gamma * cls_loss + reg_loss
#     # print('cls_loss, reg_loss : {}, {}'.format(round(cls_loss.item(), 6), round(reg_loss.item(), 6)))
#
#     if average:
#         N = gt_map.shape[0]
#         return loss_all / N
#
#     return loss_all, cls_loss, reg_loss


def loss(gt_map, pred_map, uv_map, gamma=0.2, average=True):
    pred_cls_map, pred_reg_map = pred_map[:, 0:2, :, :], pred_map[:, 2, :, :]
    gt_cls_map, gt_reg_map = gt_map[:, 0:2, :, :], gt_map[:, 2, :, :]

    gt_map_bin = torch.sum(uv_map)

    ################################################################################
    # cls loss
    loss1 = nn.BCELoss()
    pred_cls_f_map = pred_cls_map[:, 0, :, :]
    pred_cls_b_map = pred_cls_map[:, 1, :, :]
    gt_cls_f_map = gt_cls_map[:, 0, :, :]
    gt_cls_b_map = gt_cls_map[:, 1, :, :]

    cls_f_loss = loss1(pred_cls_f_map, gt_cls_f_map)
    cls_b_loss = loss1(pred_cls_b_map, gt_cls_b_map)
    # print(cls_f_loss, cls_b_loss)
    cls_loss = (cls_f_loss + cls_b_loss) * im_area / gt_map_bin

    ###################################################################################
    # reg loss
    err = 0.0
    # # print(gt_map_bin)
    reg_loss = torch.sum(torch.pow(gt_reg_map - pred_reg_map, 2)) / (gt_map_bin + err)
    # loss2 = nn.L1Loss()
    # reg_loss = 10000 * loss2(pred_reg_map, gt_reg_map)

    ###################################################################################
    # # ref loss
    # err = 0.0
    # # # print(gt_map_bin)
    # ref_loss = torch.sum(torch.pow(gt_ref_map - pred_ref_map, 2)) / (gt_map_bin + err)

    ##################################################################################
    loss_all = gamma * cls_loss + reg_loss
    # print('cls_loss, reg_loss : {}, {}'.format(round(cls_loss.item(), 6), round(reg_loss.item(), 6)))

    if average:
        N = gt_map.shape[0]
        return loss_all / N

    return loss_all, cls_loss, reg_loss


def regression_loss(gt_map, pred_map, f_map, b_map, gamma=1, average=True):
    """
    gt_map: [N, w, h]
    pred_map: [N, w, h]
    return: loss
    """
    err = 1
    # print(gt_map)
    # print(torch.sum(f_map))
    f_map_loss = torch.sum(torch.abs(gt_map-pred_map) * f_map)
    b_map_loss = torch.sum(torch.abs(gt_map-pred_map) * b_map)
    # print(f_map_loss, b_map_loss)
    gt_map_bin = torch.sum(torch.where(gt_map != 0, 1, 0))
    # print('-',gt_map_bin)
    loss = (f_map_loss + gamma*b_map_loss) / (gt_map_bin + err)
    # print(loss)
    if average:
        N = pred_map.shape[0]
        return loss / N

    return loss


