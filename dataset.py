from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, memory_format='channels_last'):
        self.root_dir = root_dir
        self.transform = transform
        self.memory_format = memory_format
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Build file list
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((
                        os.path.join(class_dir, fname),
                        self.class_to_idx[class_name]
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Use PIL for image loading
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            
        if self.transform:
            img = self.transform(img)
            
        # Ensure label is a tensor
        label = torch.tensor(label, dtype=torch.long)
            
        # Optional: Convert to channels_last format for better performance
        if self.memory_format == 'channels_last':
            img = img.contiguous(memory_format=torch.channels_last)
            
        return img, label 

class DALIDataloader(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id):
        super().__init__(batch_size, num_threads, device_id)
        self.input = ops.FileReader(
            file_root=data_dir,
            random_shuffle=True,
            num_shards=num_gpus,
            shard_id=device_id
        )
        self.decode = ops.ImageDecoder(device="mixed")
        self.resize = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        self.normalize = ops.CropMirrorNormalize(
            device="gpu",
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD
        )