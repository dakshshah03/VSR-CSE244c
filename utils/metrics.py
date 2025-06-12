import os
import torch
import numpy as np
from glob import glob
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# function that takes in video folders and outputs tensor batches
def load_and_check_videos(baseline, pred):
        
    print("collecting directories...")
    baseline_dirs = sorted([os.path.join(baseline, folder) for folder in os.listdir(baseline) if folder.isdigit()])
    pred_dirs = sorted([os.path.join(pred, folder) for folder in os.listdir(pred) if folder.isdigit()])

    # check that they are matching
    for dir1, dir2 in zip(baseline_dirs, pred_dirs):
        if os.path.basename(dir1) != os.path.basename(dir2):
            raise ValueError("The baseline and predictions don't have matching video folder labels")
    
    print("opening images...")
    baseline_videos = []
    pred_videos = []
    for dir1, dir2 in zip(baseline_dirs, pred_dirs):

        # get baseline
        baseline_video = []
        for image in os.listdir(dir1):
            baseline_frame = Image.open(os.path.join(dir1, image))
            baseline_video.append(baseline_frame)

        # get pred
        pred_video = []
        for image in os.listdir(dir2):
            pred_frame = Image.open(os.path.join(dir2, image))
            pred_video.append(pred_frame)
        
        baseline_videos.append(baseline_video)
        pred_videos.append(pred_video)

    print('converting videos to batch tensors...')
    baseline_videos = torch.tensor(np.array(baseline_videos), dtype=torch.float32)
    pred_videos = torch.tensor(np.array(pred_videos), dtype=torch.float32)
    
    print('converting tensors into the proper shape')
    baseline_videos = baseline_videos.permute(0, 4, 1, 2, 3)
    pred_videos = pred_videos.permute(0, 4, 1, 2, 3)
    
    print('checking correct batching size...')
    print(baseline_videos.shape)
    print(pred_videos.shape)

    if baseline_videos.shape != pred_videos.shape:
        raise ValueError(f"baseline_videos shape {baseline_video.shape} is not the same as pred_videos shape {pred_videos.shape}")

    return baseline_videos, pred_videos

# Structural similarity index measure
# takes in two folders of images representing video: baseline and pred
# runs ssim betweeen the two
def ssim_video(baseline, pred):

    baseline_videos, pred_videos = load_and_check_videos(baseline, pred)

    # uncomment to print min aand max
    # print("Min: ", torch.min(baseline_videos))
    # print("Max: ", torch.max(baseline_videos))

    print("Calculating ssim scores...")

    # predicted and target videos are shape (N, C, T, H, W)
    ssim_scores = []

    metric = StructuralSimilarityIndexMeasure(data_range=255.0)

    for t in range(baseline_videos.shape[2]):
        # (N, C, H, W)
        frame_pred = pred_videos[:, :, t, :, :]
        # (N, C, H, W)
        frame_target = baseline_videos[:, :, t, :, :]

        # run frame by frame ssim
        score = metric(frame_pred, frame_target)
        ssim_scores.append(score)

    # average over all frames
    avg_ssim = torch.stack(ssim_scores).mean()

    return avg_ssim, ssim_scores

# Peak signal noise ratio
# takes in two folders of images representing video: baseline and pred
# runs psnr betweeen the two
def psnr_video(baseline, pred):

    baseline_videos, pred_videos = load_and_check_videos(baseline, pred)

    # uncomment to print min aand max
    # print("Min: ", torch.min(baseline_videos))
    # print("Max: ", torch.max(baseline_videos))

    print("Calculating psnr scores...")

    # predicted and target videos are shape (N, C, T, H, W)
    ssim_scores = []

    metric = PeakSignalNoiseRatio(data_range=255.0)

    for t in range(baseline_videos.shape[2]):
        # (N, C, H, W)
        frame_pred = pred_videos[:, :, t, :, :]
        # (N, C, H, W)
        frame_target = baseline_videos[:, :, t, :, :]

        # run frame by frame ssim
        score = metric(frame_pred, frame_target)
        ssim_scores.append(score)

    # average over all frames
    avg_ssim = torch.stack(ssim_scores).mean()

    return avg_ssim, ssim_scores

# Learned Perceptual Image Patch Similarity
# takes in two folders of images representing video: baseline and pred
# runs lpips betweeen the two
def lpips_video(baseline, pred):

    baseline_videos, pred_videos = load_and_check_videos(baseline, pred)

    # uncomment to print min aand max
    # print("Min: ", torch.min(baseline_videos))
    # print("Max: ", torch.max(baseline_videos))

    print("Calculating lpips scores...")

    # predicted and target videos are shape (N, C, T, H, W)
    ssim_scores = []

    metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    for t in range(baseline_videos.shape[2]):
        # (N, C, H, W)
        frame_pred = pred_videos[:, :, t, :, :]
        # (N, C, H, W)
        frame_target = baseline_videos[:, :, t, :, :]

        # run frame by frame ssim
        score = metric(frame_pred, frame_target)
        ssim_scores.append(score)

    # average over all frames
    avg_ssim = torch.stack(ssim_scores).mean()

    return avg_ssim, ssim_scores

def tlpips_video(pred):

    print("collecting directories...")
    pred_dirs = sorted([os.path.join(pred, folder) for folder in os.listdir(pred) if folder.isdigit()])
    
    print("opening images...")
    pred_videos = []
    for dir1 in pred_dirs:

        # get pred
        pred_video = []
        for image in os.listdir(dir1):
            pred_frame = Image.open(os.path.join(dir1, image))
            pred_video.append(pred_frame)
        pred_videos.append(pred_video)

    print('converting videos to batch tensors...')
    pred_videos = torch.tensor(np.array(pred_videos), dtype=torch.float32)
    
    print('converting tensors into the proper shape')
    pred_videos = pred_videos.permute(0, 4, 1, 2, 3)
    
    print(pred_videos.shape)

    print("Calculating temporal lpips scores...")

    temporal_lpips_scores = []
    metric = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    # Loop over time (t, t+1)
    for t in range(pred_videos.shape[2] - 1):
        # Compare frame t to frame t+1
        frame_1 = pred_videos[:, :, t, :, :]
        frame_2 = pred_videos[:, :, t + 1, :, :]

        score = metric(frame_1, frame_2)
        temporal_lpips_scores.append(score)

    avg_temporal_lpips = torch.stack(temporal_lpips_scores).mean()
    return avg_temporal_lpips, temporal_lpips_scores

def main():

    print('Comparing ground truth and baseline ssim')
    avg_ssim, ssim_scores = ssim_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'baseline'))
    print(f"Mean score: {avg_ssim}")
    print(f"All scores: {ssim_scores}")
    print()

    print('Comparing ground truth and realesr ssim')
    avg_ssim, ssim_scores = ssim_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'realesr'))
    print(f"Mean score: {avg_ssim}")
    print(f"All scores: {ssim_scores}")
    print()

    print('Comparing ground truth and stablevsr ssim')
    avg_ssim, ssim_scores = ssim_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'stablevsr'))
    print(f"Mean score: {avg_ssim}")
    print(f"All scores: {ssim_scores}")
    print()

    print('Comparing ground truth and baseline psnr')
    avg_psnr, psnr_scores = psnr_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'baseline'))
    print(f"Mean score: {avg_psnr}")
    print(f"All scores: {psnr_scores}")
    print()

    print('Comparing ground truth and realesr psnr')
    avg_psnr, psnr_scores = psnr_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'realesr'))
    print(f"Mean score: {avg_psnr}")
    print(f"All scores: {psnr_scores}")
    print()

    print('Comparing ground truth and stablevsr psnr')
    avg_psnr, psnr_scores = psnr_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'stablevsr'))
    print(f"Mean score: {avg_psnr}")
    print(f"All scores: {psnr_scores}")
    print()

    print('Comparing ground truth and baseline lpips')
    avg_lpips, lpips_scores = lpips_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'baseline'))
    print(f"Mean score: {avg_lpips}")
    print(f"All scores: {lpips_scores}")
    print()

    print('Comparing ground truth and realesr lpips')
    avg_lpips, lpips_scores = lpips_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'realesr'))
    print(f"Mean score: {avg_lpips}")
    print(f"All scores: {lpips_scores}")
    print()

    print('Comparing ground truth and stablevsr lpips')
    avg_lpips, lpips_scores = lpips_video(os.path.join('data', 'benchmark_data', 'ground_truth'), os.path.join('data', 'stablevsr'))
    print(f"Mean score: {avg_lpips}")
    print(f"All scores: {lpips_scores}")
    print()

    print('Calculating baseline tlpips')
    avg_tlpips, tlpips_scores = tlpips_video(os.path.join('data', 'baseline'))
    print(f"Mean score: {avg_tlpips}")
    print(f"All scores: {tlpips_scores}")
    print()

    print('Calculating realesr tlpips')
    avg_tlpips, tlpips_scores = tlpips_video(os.path.join('data', 'realesr'))
    print(f"Mean score: {avg_tlpips}")
    print(f"All scores: {tlpips_scores}")
    print()

    print('Calculating stablevsr tlpips')
    avg_tlpips, tlpips_scores = tlpips_video( os.path.join('data', 'stablevsr'))
    print(f"Mean score: {avg_tlpips}")
    print(f"All scores: {tlpips_scores}")
    print()

if __name__ == "__main__":
    main()
