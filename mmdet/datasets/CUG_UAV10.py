from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class CUGUAV10_VOCDataset(XMLDataset):


    CLASSES = ('well', 'pool', 'prefabricated-house', 'cultivation-mesh-cage', 'quarry', 'cable-tower',
                  'vehicle', 'ship', 'landslide', 'building')

    def __init__(self, **kwargs):
        super(CUGUAV10_VOCDataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
            self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        elif 'small' in self.img_prefix:
            self.year = 2020
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')