from ffcv.fields import RGBImageField, IntField
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, RandomHorizontalFlip, NormalizeImage, RandomResizedCrop

# First, convert your dataset to FFCV format
def write_ffcv_dataset(data_dir, write_path):
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(
            max_resolution=256,
            jpeg_quality=90,
        ),
        'label': IntField()
    })
    
    writer.from_directory(data_dir)

# Create optimized loader
def create_ffcv_loader(dataset_path, batch_size, num_workers=8, distributed=False):
    image_pipeline = [
        RandomResizedCrop(224, scale=(0.08, 1.0)),
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice('cuda'),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    
    label_pipeline = [
        ToTensor(),
        ToDevice('cuda')
    ]
    
    loader = Loader(
        dataset_path,
        batch_size=batch_size,
        num_workers=num_workers,
        order=OrderOption.RANDOM,
        distributed=distributed,
        pipelines={
            'image': image_pipeline,
            'label': label_pipeline
        }
    )
    return loader 