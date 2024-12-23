from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torchvision.models import resnet50
from tqdm import tqdm
import time
import os
from torch.cuda.amp import autocast, GradScaler
import random
import shutil

# Standard ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_imagenet_transforms(image_size):
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(int(image_size*1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    
    return train_transforms, val_transforms

def get_data_loaders(data_dir, batch_size=512, subset_fraction=0.1, image_size=160):
    train_transforms, val_transforms = get_imagenet_transforms(image_size)
    
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
    )
    total_samples = len(train_dataset)
    subset_size = int(total_samples * subset_fraction)
    
    indices = torch.randperm(total_samples)[:subset_size]
    subset_sampler = torch.utils.data.SubsetRandomSampler(indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=subset_sampler,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transforms
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    return train_loader, val_loader

def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch):
    model.train()
    torch.cuda.empty_cache()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    accumulation_steps = 4
    optimizer.zero_grad(set_to_none=True)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/40')
    for i, (images, labels) in enumerate(pbar):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        
        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({
            'loss': running_loss/total,
            'acc': 100.*correct/total
        })
    
    return running_loss/len(train_loader), 100.*correct/total

def validate(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation Epoch {epoch+1}/40')
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': running_loss/total,
                'acc': 100.*correct/total
            })
    
    return running_loss/len(val_loader), 100.*correct/total

def create_subset_dataset(data_dir, subset_fraction=0.1):
    """Create a balanced subset of ImageNet by taking subset_fraction from each class"""
    subset_dir = data_dir + '_subset'
    
    # Create subset directory
    for split in ['train', 'val']:
        source_dir = os.path.join(data_dir, split)
        target_dir = os.path.join(subset_dir, split)
        
        # For each class directory
        for class_name in os.listdir(source_dir):
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
            num_subset = int(num_images * subset_fraction)
            selected_images = random.sample(images, num_subset)
            
            # Copy selected images
            for img_name in selected_images:
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(target_class_dir, img_name)
                shutil.copy2(src, dst)
                
    return subset_dir

def main():
    data_dir = '/path/to/imagenet'
    
    # Create subset first time
    if not os.path.exists(data_dir + '_subset'):
        print("Creating 10% subset of ImageNet...")
        data_dir = create_subset_dataset(data_dir, subset_fraction=0.1)
        print(f"Subset created at {data_dir}")
    else:
        data_dir = data_dir + '_subset'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model setup
    model = resnet50(num_classes=1000)
    model = model.to(device)
    model = model.to(memory_format=torch.channels_last)  # Optimize memory layout
    
    # Training parameters
    learning_rate = 0.1
    
    # Loss and optimizer setup
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
        weight_decay=1e-4
    )
    
    best_val_acc = 0.0
    
    # Phase 1: Fast initial training with smaller images
    print("Starting Phase 1: Training with smaller images...")
    image_size = 160
    batch_size = 1024
    
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        subset_fraction=0.1,
        image_size=image_size
    )
    
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )
    
    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch
        )
        
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
            print(f"Phase 1 - Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, 'phase1_best_model.pth')
    
    # Phase 2: Fine-tune with larger images
    print("\nStarting Phase 2: Fine-tuning with larger images...")
    image_size = 224
    batch_size = 512
    
    train_loader, val_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        subset_fraction=0.1,
        image_size=image_size
    )
    
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate/10,  # Lower learning rate for fine-tuning
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1e4
    )
    
    for epoch in range(20):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, scaler, epoch
        )
        
        if (epoch + 1) % 5 == 0:
            val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)
            print(f"Phase 2 - Epoch {epoch+1}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                }, 'final_best_model.pth')
    
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == '__main__':
    main() 