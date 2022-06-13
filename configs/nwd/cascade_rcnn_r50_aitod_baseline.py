"""
CascadeRPN

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.133
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.335
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.078
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.039
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.129
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.181
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.263
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.256
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.279
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.284
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.093
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.291
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.357
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.431
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.881
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.301
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.502
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.638
# Class-specific LRP-Optimal Thresholds # 
 [0.731 0.831 0.769 0.809 0.582 0.646 0.59  0.465]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.133 | bridge        | 0.051 | storage-tank | 0.291 |
| ship     | 0.248 | swimming-pool | 0.033 | vehicle      | 0.214 |
| person   | 0.064 | wind-mill     | 0.026 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.885 | bridge        | 0.945 | storage-tank | 0.746 |
| ship     | 0.793 | swimming-pool | 0.951 | vehicle      | 0.817 |
| person   | 0.940 | wind-mill     | 0.968 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
"""

_base_ = '../faster_rcnn/faster_rcnn_r50_caffe_fpn_1x_aitod.py'
rpn_weight = 0.7
model = dict(
    rpn_head=dict(
        _delete_=True,
        type='CascadeRPNHead',
        num_stages=2,
        stages=[
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[1.0],
                    strides=[4, 8, 16, 32, 64]),
                adapt_cfg=dict(type='dilation', dilation=3),
                bridged_feature=True,
                sampling=False,
                with_cls=False,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.1, 0.1, 0.5, 0.5)),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight)),
            dict(
                type='StageCascadeRPNHead',
                in_channels=256,
                feat_channels=256,
                adapt_cfg=dict(type='offset'),
                bridged_feature=False,
                sampling=True,
                with_cls=True,
                reg_decoded_bbox=True,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=(.0, .0, .0, .0),
                    target_stds=(0.05, 0.05, 0.1, 0.1)),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=True,
                    loss_weight=1.0 * rpn_weight),
                loss_bbox=dict(
                    type='IoULoss', linear=True,
                    loss_weight=10.0 * rpn_weight))
        ]),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(target_stds=[0.04, 0.04, 0.08, 0.08]),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.5),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=[
            dict(
                assigner=dict(
                    type='RegionAssigner', center_ratio=0.2, ignore_ratio=0.5),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.3,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False),
                allowed_border=-1,
                pos_weight=-1,
                debug=False)
        ],
        rpn_proposal=dict(max_per_img=3000, nms=dict(iou_threshold=0.8)),
        rcnn=dict(
            assigner=dict(
                pos_iou_thr=0.65, neg_iou_thr=0.65, min_pos_iou=0.65),
            sampler=dict(type='RandomSampler', num=256))),
    test_cfg=dict(
        rpn=dict(max_per_img=3000, nms=dict(iou_threshold=0.8)),
        rcnn=dict(score_thr=1e-3)))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
