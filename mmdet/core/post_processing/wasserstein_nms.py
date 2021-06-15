import os
import sys

import numpy as np
import torch

from mmcv.utils import deprecated_api_warning


def wasserstein_nms_op(dets_wl, scores, order, dets_sorted, iou_threshold, multi_label):
    pass

def wasserstein_nms(dets, scores, iou_threshold, labels=None):
    """Performs non-maximum suppression (NMS) on the rotated boxes according to
    their intersection-over-union (IoU).

    Rotated NMS iteratively removes lower scoring rotated boxes which have an
    IoU greater than iou_threshold with another (higher scoring) rotated box.

    Args:
        boxes (Tensor):  Rotated boxes in shape (N, 5). They are expected to \
            be in (x_ctr, y_ctr, width, height, angle_radian) format.
        scores (Tensor): scores in shape (N, ).
        iou_threshold (float): IoU thresh for NMS.
        labels (Tensor): boxes's label in shape (N,).

    Returns:
        tuple: kept dets(boxes and scores) and indice, which is always the \
            same data type as the input.
    """
    if dets.shape[0] == 0:
        return dets, None
    multi_label = labels is not None
    if multi_label:
        dets_wl = torch.cat((dets, labels.unsqueeze(1)), 1)
    else:
        dets_wl = dets
    _, order = scores.sort(0, descending=True)
    dets_sorted = dets_wl.index_select(0, order)

    keep_inds = wasserstein_nms_op(dets_wl, scores, order, dets_sorted,
                                           iou_threshold, multi_label)
    dets = torch.cat((dets[keep_inds], scores[keep_inds].reshape(-1, 1)),
                     dim=1)
    return dets, keep_inds

def batched_wasserstein_nms(boxes, scores, idxs, nms_cfg, class_agnostic=False):
    """Performs non-maximum suppression in a batched fashion.

    Modified from https://github.com/pytorch/vision/blob
    /505cd6957711af790211896d32b40291bea1bc21/torchvision/ops/boxes.py#L39.
    In order to perform NMS independently per class, we add an offset to all
    the boxes. The offset is dependent only on the class idx, and is large
    enough so that boxes from different classes do not overlap.

    Arguments:
        boxes (torch.Tensor): boxes in shape (N, 4).
        scores (torch.Tensor): scores in shape (N, ).
        idxs (torch.Tensor): each index value correspond to a bbox cluster,
            and NMS will not be applied between elements of different idxs,
            shape (N, ).
        nms_cfg (dict): specify nms type and other parameters like iou_thr.
            Possible keys includes the following.

            - iou_thr (float): IoU threshold used for NMS.
            - split_thr (float): threshold number of boxes. In some cases the
                number of boxes is large (e.g., 200k). To avoid OOM during
                training, the users could set `split_thr` to a small value.
                If the number of boxes is greater than the threshold, it will
                perform NMS on each group of boxes separately and sequentially.
                Defaults to 10000.
        class_agnostic (bool): if true, nms is class agnostic,
            i.e. IoU thresholding happens over all boxes,
            regardless of the predicted class.

    Returns:
        tuple: kept dets and indice.
    """
    nms_cfg_ = nms_cfg.copy()
    class_agnostic = nms_cfg_.pop('class_agnostic', class_agnostic)
    if class_agnostic:
        boxes_for_nms = boxes
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

    nms_type = nms_cfg_.pop('type', 'nms')
    nms_op = eval(nms_type)

    split_thr = nms_cfg_.pop('split_thr', 10000)
    # Won't split to multiple nms nodes when exporting to onnx
    if boxes_for_nms.shape[0] < split_thr or torch.onnx.is_in_onnx_export():
        dets, keep = nms_op(boxes_for_nms, scores, **nms_cfg_)
        boxes = boxes[keep]
        # -1 indexing works abnormal in TensorRT
        # This assumes `dets` has 5 dimensions where
        # the last dimension is score.
        # TODO: more elegant way to handle the dimension issue.
        scores = dets[:, 4]
    else:
        total_mask = scores.new_zeros(scores.size(), dtype=torch.bool)
        for id in torch.unique(idxs):
            mask = (idxs == id).nonzero(as_tuple=False).view(-1)
            dets, keep = nms_op(boxes_for_nms[mask], scores[mask], **nms_cfg_)
            total_mask[mask[keep]] = True

        keep = total_mask.nonzero(as_tuple=False).view(-1)
        keep = keep[scores[keep].argsort(descending=True)]
        boxes = boxes[keep]
        scores = scores[keep]

    return torch.cat([boxes, scores[:, None]], -1), keep