import torch
import torch.nn as nn
from torchvision.models import resnet50
from ffcv.fields import RGBImageField, IntField
from ffcv.writer import DatasetWriter
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, RandomHorizontalFlip, NormalizeImage, RandomResizedCrop
from tqdm import tqdm
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets
from PIL import Image
import random
import shutil

# Constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255

def write_ffcv_dataset(data_dir, write_path):
    """Convert ImageNet dataset to FFCV format"""
    dataset = datasets.ImageFolder(data_dir)
    total_samples = len(dataset)
    
    if total_samples == 0:
        raise ValueError(f"No samples found in {data_dir}")
    
    print(f"Found {total_samples} samples in {data_dir}")
    
    # First pass: collect image dimensions
    print("Collecting image dimensions...")
    max_width = 0
    max_height = 0
    
    for idx in tqdm(range(total_samples), desc="Scanning images"):
        img_path, _ = dataset.samples[idx]
        with Image.open(img_path) as img:
            width, height = img.size
            max_width = max(max_width, width)
            max_height = max(max_height, height)
    
    print(f"Max dimensions: {max_width}x{max_height}")
    
    # Create writer with max dimensions
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(
            max_resolution=max(max_width, max_height),
            jpeg_quality=90,
        ),
        'label': IntField()
    }, num_workers=4)
    
    # Write samples
    images = []
    labels = []
    
    for idx in tqdm(range(total_samples), desc="Loading dataset"):
        img_path, label = dataset.samples[idx]
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            continue
    
    if not images:
        raise ValueError("No valid images found")
    
    print(f"Writing {len(images)} samples to FFCV format...")
    writer.from_indexed_dataset({
        'image': images,
        'label': labels
    })
    
    print(f"Successfully wrote {len(images)} samples to {write_path}")

def create_ffcv_loader(dataset_path, batch_size, is_train=True, num_workers=4):
    """Create FFCV data loader"""
    # Start with a simple pipeline for both train and val
    image_pipeline = [
        ToTensor(),
        ToDevice('cuda'),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]
    
    label_pipeline = [ToTensor(), ToDevice('cuda')]
    
    try:
        loader = Loader(
            dataset_path,
            batch_size=batch_size,
            num_workers=num_workers,
            order=OrderOption.RANDOM if is_train else OrderOption.SEQUENTIAL,
            pipelines={
                'image': image_pipeline,
                'label': label_pipeline
            },
            os_cache=True
        )
        # Test the loader
        next(iter(loader))
        return loader
    except Exception as e:
        print(f"Error creating loader: {e}")
        print(f"Dataset path: {dataset_path}")
        print(f"Dataset exists: {os.path.exists(dataset_path)}")
        if os.path.exists(dataset_path):
            print(f"Dataset size: {os.path.getsize(dataset_path)}")
            # Try to read the first few bytes
            with open(dataset_path, 'rb') as f:
                print(f"First 100 bytes: {f.read(100)}")
        raise

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}')
    for i, (images, labels) in enumerate(pbar):
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        if scheduler is not None:
            scheduler.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss/(i+1),
            'acc': 100.*correct/total
        })
    
    return running_loss/len(loader), 100.*correct/total

def validate(model, loader, criterion):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Validation')
        for i, (images, labels) in enumerate(pbar):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/(i+1),
                'acc': 100.*correct/total
            })
    
    return running_loss/len(loader), 100.*correct/total

def create_subset_dataset(data_dir, subset_fraction=0.1):
    """Create a balanced subset of ImageNet"""
    subset_dir = os.path.join('/kaggle/working', 'imagenet_subset')
    
    # Create subset directory
    for split in ['train', 'val']:
        source_dir = os.path.join(data_dir, split)
        target_dir = os.path.join(subset_dir, split)
        
        print(f"Creating {split} subset...")
        # For each class directory
        for class_name in tqdm(os.listdir(source_dir)):
            class_dir = os.path.join(source_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            # Create class directory in subset
            target_class_dir = os.path.join(target_dir, class_name)
            os.makedirs(target_class_dir, exist_ok=True)
            
            # List all images in the class
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            
            # Select random subset
            num_images = len(images)
            num_subset = max(1, int(num_images * subset_fraction))
            selected_images = random.sample(images, num_subset)
            
            # Copy selected images
            for img_name in selected_images:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(target_class_dir, img_name)
                shutil.copy2(src, dst)
    
    return subset_dir

def main():
    # Verify dataset structure
    data_dir = '/kaggle/working/imagenet-mini'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    print(f"Checking dataset structure...")
    print(f"Train dir exists: {os.path.exists(train_dir)}")
    print(f"Val dir exists: {os.path.exists(val_dir)}")
    
    if os.path.exists(train_dir):
        print(f"Train classes: {len(os.listdir(train_dir))}")
    if os.path.exists(val_dir):
        print(f"Val classes: {len(os.listdir(val_dir))}")
    
    # Kaggle paths for manually downloaded ImageNet-mini
    data_dir = '/kaggle/working/imagenet-mini'  # Path to your uploaded dataset
    ffcv_dir = '/kaggle/working/ffcv'
    os.makedirs(ffcv_dir, exist_ok=True)
    
    # Convert to FFCV format
    train_ffcv = os.path.join(ffcv_dir, 'train.beton')
    val_ffcv = os.path.join(ffcv_dir, 'val.beton')
    
    if not os.path.exists(train_ffcv):
        write_ffcv_dataset(os.path.join(data_dir, 'train'), train_ffcv)
    if not os.path.exists(val_ffcv):
        write_ffcv_dataset(os.path.join(data_dir, 'val'), val_ffcv)
    
    # Training parameters
    batch_size = 512  # Adjust based on GPU memory
    num_epochs = 40
    learning_rate = 0.1
    
    # Create data loaders
    train_loader = create_ffcv_loader(train_ffcv, batch_size, is_train=True)
    val_loader = create_ffcv_loader(val_ffcv, batch_size, is_train=False)
    
    # Model setup
    model = resnet50(num_classes=1000)
    model = model.to('cuda')
    model = model.to(memory_format=torch.channels_last)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    # Learning rate schedule
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )
    
    scaler = GradScaler()
    best_acc = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, epoch
        )
        
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = validate(model, val_loader, criterion)
            print(f"\nEpoch {epoch+1}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, '/kaggle/working/best_model.pth')
    
    print(f"Training completed! Best accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main() 