from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator

@pipeline_def
def create_dali_pipeline(data_dir, batch_size, num_threads, device_id):
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
        num_shards=1,
        shard_id=0,
        name="Reader"
    )
    
    images = fn.decoders.image(
        jpegs,
        device="mixed",
        output_type=types.RGB
    )
    
    images = fn.resize(
        images,
        device="gpu",
        resize_x=224,
        resize_y=224,
        interp_type=types.INTERP_LINEAR
    )
    
    images = fn.crop_mirror_normalize(
        images,
        device="gpu",
        dtype=types.FLOAT,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_layout="CHW"
    )
    
    return images, labels 