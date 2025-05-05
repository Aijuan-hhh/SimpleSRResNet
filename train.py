import os
import argparse
import numpy as np
from PIL import Image
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

from SRResNet import SimpleSRResNet

class BSDSDataset(Dataset):
    """
    Dataset for BSDS500: loads high-resolution images, crops random patches,
    and creates low-resolution images by downsampling (bicubic) by scale factor.
    """
    def __init__(self, root_dir, split='train', scale_factor=2, patch_size=48):
        self.scale = scale_factor
        self.patch_size = patch_size
        self.hr_patch = patch_size * scale_factor

        images_dir = os.path.join(root_dir, 'images')
        base_dir = images_dir if os.path.isdir(images_dir) else root_dir
        self.split_dir = os.path.join(base_dir, split)
        if not os.path.isdir(self.split_dir):
            raise ValueError(f"Directory not found: {self.split_dir}")
        self.image_paths = [os.path.join(self.split_dir, f)
                            for f in sorted(os.listdir(self.split_dir))
                            if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
        if not self.image_paths:
            raise ValueError(f"No images found in {self.split_dir}")
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        hr = Image.open(img_path).convert('RGB')
        w, h = hr.size
        # Ensure image is large enough
        if w < self.hr_patch or h < self.hr_patch:
            # resize up to at least hr_patch
            hr = hr.resize((max(w, self.hr_patch), max(h, self.hr_patch)), Image.BICUBIC)
            w, h = hr.size

        # Random crop HR patch
        left = random.randint(0, w - self.hr_patch)
        top = random.randint(0, h - self.hr_patch)
        hr_patch = hr.crop((left, top, left + self.hr_patch, top + self.hr_patch))

        # Create low-resolution patch
        lr_patch = hr_patch.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        # 确保数据转换为 Tensor 后范围是 [0,1]
        lr_tensor = self.to_tensor(lr_patch).float()
        hr_tensor = self.to_tensor(hr_patch).float()
        return lr_tensor, hr_tensor

class AdamOptimizer:
    """Simple NumPy-based Adam optimizer"""
    def __init__(self, params, lr=1e-4, betas=(0.9,0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k,v in params.items()}
        self.v = {k: np.zeros_like(v) for k,v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for k in self.params:
            g = grads.get(k)
            if g is None: continue
            self.m[k] = self.beta1*self.m[k] + (1-self.beta1)*g
            self.v[k] = self.beta2*self.v[k] + (1-self.beta2)*(g*g)
            m_hat = self.m[k] / (1 - self.beta1**self.t)
            v_hat = self.v[k] / (1 - self.beta2**self.t)
            self.params[k] -= lr_t * m_hat / (np.sqrt(v_hat) + self.eps)


def train(args):
    # Prepare data loader
    dataset = BSDSDataset(root_dir=args.data_dir, split='train', scale_factor=args.scale, patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # Initialize model
    model = SimpleSRResNet(scale_factor=args.scale)
    model.initialize((3,args.patch_size,args.patch_size))

    # Set up optimizer
    params, _ = model.get_params_and_grads()
    optimizer = AdamOptimizer(params, lr=args.lr)

    # Lists for logging
    loss_list = []
    psnr_list = []

    # Training loop
    for epoch in range(1, args.epochs+1):
        total_loss = 0.0
        for lr_batch, hr_batch in loader:
            lr_np, hr_np = lr_batch.numpy(), hr_batch.numpy()
            # Forward
            pred = model.forward(lr_np, training=True)
            # MSE损失
            loss = np.mean((pred - hr_np)**2)
            total_loss += loss
            # Backward
            grad_out = (2.0 / pred.size) * (pred - hr_np)
            model.backward(grad_out)
            # Update
            _, grads = model.get_params_and_grads()
            optimizer.step(grads)
        avg_loss = total_loss / len(loader)
        # Compute approximate PSNR
        # psnr = 10 * np.log10(1.0 / avg_loss) if avg_loss > 0 else 0
        # 添加极小值 1e-10 防止数值溢出
        psnr = 10 * np.log10(1.0 / (avg_loss + 1e-10))
        loss_list.append(avg_loss)
        psnr_list.append(psnr)
        print(f"Epoch {epoch}/{args.epochs} - Loss: {avg_loss:.6f} - PSNR: {psnr:.2f} dB")

    # Save checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.save_name)
    model.save_weights(ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    # Plot curves
    epochs = list(range(1, args.epochs+1))
    # Loss curve
    plt.figure()
    plt.plot(epochs, loss_list)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.savefig(os.path.join(args.save_dir, 'loss_curve.png'))
    # PSNR curve

    plt.figure()
    plt.plot(epochs, psnr_list)
    plt.title('Training PSNR')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.savefig(os.path.join(args.save_dir, 'psnr_curve.png'))
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser("Train SRResNet with Adam on BSDS500")
    p.add_argument("--data_dir", type=str, default="./BSD500")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=48, help="Low-res patch size; HR patch is patch_size*scale")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--save_name", type=str, default="best_model.npz")
    args = p.parse_args()
    train(args)