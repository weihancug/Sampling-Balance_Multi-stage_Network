from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class NWPU_VOCDataset(XMLDataset):

    CLASSES = ('airplane', 'ship', 'storage tank', 'baseball diamond', 'tennis court', 'basketball court', 'ground track field',
               'habor', 'bridge', 'vehicle')


    def __init__(self, **kwargs):
        super(NWPU_VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        elif 'NWPU' in self.img_prefix:
            self.year = 2017
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')