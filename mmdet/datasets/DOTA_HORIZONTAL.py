from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class DOTA_HORIZONTAL_VOCDataset(XMLDataset):

    CLASSES = ('small-vehicle', 'plane', 'large-vehicle', 'ship', 'harbor', 'tennis-court',
                  'ground-track-field', 'soccer-ball-field', 'baseball-diamond', 'swimming-pool', 'roundabout',
                  'basketball-court', 'storage-tank', 'bridge', 'helicopter')

    def __init__(self, **kwargs):
        super(DOTA_HORIZONTAL_VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        elif 'NWPU' in self.img_prefix:
            self.year = 2017
        elif 'DOTA' in self.img_prefix:
            self.year = 2018
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')