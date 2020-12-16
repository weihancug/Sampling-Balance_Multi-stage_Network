from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class DOIR_Dataset(XMLDataset):

    CLASSES = ('harbor', 'vehicle', 'ship', 'Expressway-toll-station', 'bridge', 'baseballfield', 'basketballcourt', 'overpass',
                'chimney', 'trainstation', 'storagetank', 'dam', 'tenniscourt', 'groundtrackfield', 'stadium', 'windmill',
               'airport', 'Expressway-Service-area', 'airplane', 'golffield')

    def __init__(self, **kwargs):
        super(DOIR_Dataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
         self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
