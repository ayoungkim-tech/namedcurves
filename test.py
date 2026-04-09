import argparse
from models.model import NamedCurves
import torch
import os
from omegaconf import OmegaConf
from glob import glob
from PIL import Image
from torchvision.transforms import functional as TF
import numpy as np
import matplotlib.pyplot as plt
import lpips
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='assets/a4957-input.png')
    parser.add_argument('--output_path', type=str, default='output/')
    parser.add_argument('--model_path', type=str, default='pretrained/mit5k_uegan_psnr_25.59.pth')
    parser.add_argument('--config_path', type=str, default='configs/mit5k_dpe_config.yaml')
    return parser.parse_args()

# Test LPIPS
loss_fn = lpips.LPIPS(net='alex').cuda()

def main():
    args = parse_args()
    config = OmegaConf.load(args.config_path)
    model = NamedCurves(config.model).cuda()
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    #check if input_path is a folder
    if os.path.isdir(args.input_path):
        input_paths = glob(sorted(args.input_path + '/*'))
    
    else:
        input_paths = [args.input_path]
    
    for input_path in input_paths:
        input_tensor = TF.to_tensor(Image.open(input_path)).unsqueeze(0)
        output = model(input_tensor.cuda())
        output = TF.to_pil_image(output[0].cpu())

        # ----------------------- DEBUG -----------------------
        input_image = Image.open(input_path)
        input_np = np.array(input_image)
        output_np = np.array(output)
        # --- LPIPS ---
        input_lpips = TF.to_tensor(input_image).unsqueeze(0).cuda()
        output_lpips = TF.to_tensor(output).unsqueeze(0).cuda()
        with torch.no_grad():
            lpips_value = loss_fn(input_lpips, output_lpips)
        print("LPIPS:", lpips_value.item())
        # --- SSIM ---
        ssim_value = ssim(input_np, output_np, channel_axis=2, data_range=255)
        print(f"SSIM: {ssim_value:.4f}")
        # --- PSNR ---
        psnr_value = psnr(input_np, output_np, data_range=255)
        print(f"PSNR: {psnr_value:.2f} dB")
        # --- PLOT ---
        plt.subplot(1,2,1)
        plt.title("Input")
        plt.imshow(input_np)
        plt.subplot(1,2,2)
        plt.title("Output")
        plt.imshow(output_np)
        plt.show()
        # ---------------------------------------------- 

        output.save(os.path.join(args.output_path, os.path.basename(input_path)))

if __name__ == '__main__':
    main()