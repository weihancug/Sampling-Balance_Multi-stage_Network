from .registry import DATASETS
from .xml_style import XMLDataset


@DATASETS.register_module
class HRRSD_Dataset(XMLDataset):

    CLASSES = ('airplane', 'ship', 'storage tank', 'tennis court','basketball court', 'ground track field', 'harbor',
               'bridge','vehicle','baseball diamond','crossroad','T junction','parking lot')

    def __init__(self, **kwargs):
        super(HRRSD_Dataset, self).__init__(**kwargs)
        if 'VOC2007' in self.img_prefix:
         self.year = 2007
        elif 'VOC2012' in self.img_prefix:
            self.year = 2012
        else:
            raise ValueError('Cannot infer dataset year from img_prefix')
