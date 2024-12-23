from ffcv.fields import RGBImageField, IntField
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Convert

def create_ffcv_loader(dataset_path, batch_size):
    loaders = Loader(
        dataset_path,
        batch_size=batch_size,
        num_workers=8,
        order=OrderOption.RANDOM,
        pipelines={
            'image': [ToTensor(), ToDevice('cuda'), Convert('float32')],
            'label': [ToTensor(), ToDevice('cuda')]
        }
    )
    return loaders 