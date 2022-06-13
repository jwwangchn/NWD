"""

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.178
Average Precision  (AP) @[ IoU=0.25      | area=   all | maxDets=1500 ] = -1.000
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.438
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=1500 ] = 0.110
Average Precision  (AP) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.025
Average Precision  (AP) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.170
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.261
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.344
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.284
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=300 ] = 0.302
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=1500 ] = 0.305
Average Recall     (AR) @[ IoU=0.50:0.95 | area=verytiny | maxDets=1500 ] = 0.023
Average Recall     (AR) @[ IoU=0.50:0.95 | area=  tiny | maxDets=1500 ] = 0.315
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1500 ] = 0.400
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1500 ] = 0.459
Optimal LRP             @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.844
Optimal LRP Loc         @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.293
Optimal LRP FP          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.419
Optimal LRP FN          @[ IoU=0.50      | area=   all | maxDets=1500 ] = 0.551
# Class-specific LRP-Optimal Thresholds # 
 [0.629 0.777 0.714 0.819 0.668 0.667 0.6   0.581]

+----------+-------+---------------+-------+--------------+-------+
| category | AP    | category      | AP    | category     | AP    |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.256 | bridge        | 0.114 | storage-tank | 0.306 |
| ship     | 0.305 | swimming-pool | 0.098 | vehicle      | 0.228 |
| person   | 0.080 | wind-mill     | 0.037 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+

+----------+-------+---------------+-------+--------------+-------+
| category | oLRP  | category      | oLRP  | category     | oLRP  |
+----------+-------+---------------+-------+--------------+-------+
| airplane | 0.784 | bridge        | 0.902 | storage-tank | 0.726 |
| ship     | 0.736 | swimming-pool | 0.909 | vehicle      | 0.801 |
| person   | 0.926 | wind-mill     | 0.968 | None         | None  |
+----------+-------+---------------+-------+--------------+-------+
"""
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_aitod.py',
    '../_base_/datasets/aitod_detection.py',
    '../_base_/schedules/schedule_1x.py', 
    '../_base_/default_runtime.py'
]


model = dict(
    rpn_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='WassersteinLoss', loss_weight=10.0)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                gpu_assign_thr=512)),
        rpn_proposal=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85)),
        rpn=dict(
            assigner=dict(
                gpu_assign_thr=512, 
                iou_calculator=dict(type='BboxDistanceMetric'),
                assign_metric='wasserstein'))),
    test_cfg=dict(
        rpn=dict(
            nms_pre=3000,
            max_per_img=3000,
            nms=dict(type='wasserstein_nms', iou_threshold=0.85),
            min_bbox_size=0)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

fp16 = dict(loss_scale=512.)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# learning policy
checkpoint_config = dict(interval=4)
evaluation = dict(interval=12, metric='bbox')
