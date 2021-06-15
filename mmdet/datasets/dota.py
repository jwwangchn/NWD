from .builder import DATASETS
from .coco import CocoDataset
from .aitod import AITODDataset

import itertools
import logging
from collections import OrderedDict

import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class DOTA2Dataset(AITODDataset):

    CLASSES = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane', 'airport', 'helipad')