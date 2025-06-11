import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import argparse
import RRDBNet_arch as arch

def main(args):
    model_path = args.model_path
    device = torch.device(args.device)

    model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    print(f'Model path {model_path}. \nTesting...')

    input_pattern = osp.join(args.input_dir, '*')
    os.makedirs(args.output_dir, exist_ok=True)

    for idx, path in enumerate(glob.glob(input_pattern), start=1):
        base = osp.splitext(osp.basename(path))[0]
        print(idx, base)

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"Warning: Failed to read {path}")
            continue

        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()

        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        out_path = osp.join(args.output_dir, f'{base}_rlt.png')
        cv2.imwrite(out_path, output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Super-resolve images using ESRGAN.')
    parser.add_argument('--input_dir', type=str, required=True, help='Input image directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output image directory')
    parser.add_argument('--model_path', type=str, default='./../submodules/ESRGAN/models/RRDB_ESRGAN_x4.pth', help='Path to the ESRGAN model')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation')

    args = parser.parse_args()
    main(args)
