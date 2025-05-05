import os
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt  # 新增导入

from SRResNet import SimpleSRResNet

def compute_psnr(pred, target, max_val=1.0):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)

class BSDSDataset(Dataset):
    """裁剪小块用于测试，保证和训练一致"""

    def __init__(self, root_dir, split='test', scale_factor=2, patch_size=48):
        self.scale = scale_factor
        self.lr_patch = patch_size
        self.hr_patch = patch_size * scale_factor

        images_dir = os.path.join(root_dir, 'images') if os.path.isdir(os.path.join(root_dir, 'images')) else root_dir
        self.split_dir = os.path.join(images_dir, split)
        if not os.path.isdir(self.split_dir):
            raise ValueError(f"Directory not found: {self.split_dir}")

        self.image_paths = [os.path.join(self.split_dir, f)
                            for f in sorted(os.listdir(self.split_dir))
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        if not self.image_paths:
            raise ValueError(f"No images found in {self.split_dir}")

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        hr_img = Image.open(img_path).convert('RGB')
        w, h = hr_img.size

        # 确保可以裁剪
        if w < self.hr_patch or h < self.hr_patch:
            hr_img = hr_img.resize((max(w, self.hr_patch), max(h, self.hr_patch)), Image.BICUBIC)
            w, h = hr_img.size

        # 固定中心裁剪（测试阶段）
        left = (w - self.hr_patch) // 2
        top = (h - self.hr_patch) // 2
        hr_patch = hr_img.crop((left, top, left + self.hr_patch, top + self.hr_patch))

        # 下采样得到 LR patch
        lr_patch = hr_patch.resize((self.lr_patch, self.lr_patch), Image.BICUBIC)

        return self.to_tensor(lr_patch), self.to_tensor(hr_patch), img_path  # 返回原图路径

def test(args):
    # 创建结果保存目录
    save_dir = os.path.join("./test_results")
    os.makedirs(save_dir, exist_ok=True)

    # Dataset
    dataset = BSDSDataset(root_dir=args.data_dir, split='test', scale_factor=args.scale,
                                patch_size=args.patch_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Model
    model = SimpleSRResNet(scale_factor=args.scale)
    model.initialize((3, args.patch_size, args.patch_size))
    model.load_weights(args.model_path)

    total_psnr = 0.0
    num_images = 0

    pbar = tqdm(loader, desc="Testing patches", unit="patch")

    # 标记是否已保存示例图像
    example_saved = False

    for lr_batch, hr_batch, img_paths in pbar:  # 接收原图路径
        lr_np = lr_batch.numpy()
        hr_np = hr_batch.numpy()

        # Forward
        with np.errstate(over='ignore'):
            pred_np = model.forward(lr_np, training=False)

        pred_np = np.clip(pred_np, 0.0, 1.0)

        psnr = compute_psnr(pred_np[0], hr_np[0])
        total_psnr += psnr
        num_images += 1

        pbar.set_postfix(psnr=f"{psnr:.2f} dB")

        # 保存测试结果图（每张原图保存一次）
        base_name = os.path.splitext(os.path.basename(img_paths[0]))[0]
        save_path = os.path.join(save_dir, f"{base_name}_result.png")

        # 转换张量为PIL图像
        lr_img = transforms.ToPILImage()(lr_batch.squeeze(0))
        sr_img = transforms.ToPILImage()(torch.from_numpy(pred_np[0]))
        hr_img = transforms.ToPILImage()(hr_batch.squeeze(0))

        # 保存结果图（LR, SR, HR三合一）
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(lr_img)
        axes[0].set_title("Low-Resolution (Input)")
        axes[0].axis('off')

        axes[1].imshow(sr_img)
        axes[1].set_title(f"Super-Resolution (PSNR: {psnr:.2f} dB)")
        axes[1].axis('off')

        axes[2].imshow(hr_img)
        axes[2].set_title("High-Resolution (Ground Truth)")
        axes[2].axis('off')

        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

        # 仅保存第一个样本的对比图用于展示
        if not example_saved:
            example_saved = True
            # 显示图像（可选）
            plt.figure(figsize=(15, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(lr_img)
            plt.title("Low-Resolution (Input)")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(sr_img)
            plt.title(f"Super-Resolution (PSNR: {psnr:.2f} dB)")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(hr_img)
            plt.title("High-Resolution (Ground Truth)")
            plt.axis('off')

            plt.tight_layout()
            plt.show()

    avg_psnr = total_psnr / num_images
    print(f"\nTested {num_images} patches — Average PSNR: {avg_psnr:.2f} dB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test SRResNet on patches extracted from BSDS500")
    parser.add_argument("--data_dir", type=str, default="./BSD500", help="Dataset path")
    parser.add_argument("--model_path", type=str, default="./checkpoints/best_model.npz",
                        help="Model checkpoint (.npz)")
    parser.add_argument("--scale", type=int, default=2, help="Upscale factor")
    parser.add_argument("--patch_size", type=int, default=48, help="LR patch size (same as training)")
    args = parser.parse_args()
    test(args)