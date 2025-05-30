import opencv as cv2
from skimage.metrics import structural_similarity
from glob import glob
import lpips

def video_ssim(video1_dir, video2_dir):
    total_ssim = 0
    total_frames = len(glob(video1_dir + '/*.jpg'))  # should be the same for both videos, TODO: check for consistency
    
    for img1_path, img2_path in zip(sorted(glob(video1_dir + '/*.jpg')), sorted(glob(video2_dir + '/*.jpg'))):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        score = structural_similarity(img1, img2)
        total_ssim += score
        
    return total_ssim / total_frames

def video_lpips(video1_dir, video2_dir):
    total_lpips = 0
    total_frames = len(glob(video1_dir + '/*.jpg'))  # should be the same for both videos, TODO: check for consistency
    
    loss_fn = lpips.LPIPS(net='alex')  # choose network type
    
    for img1_path, img2_path in zip(sorted(glob(video1_dir + '/*.jpg')), sorted(glob(video2_dir + '/*.jpg'))):
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        
        score = loss_fn.forward(img1, img2).item()
        total_lpips += score
        
    return total_lpips / total_frames

def temporal_lpips(video_dir):
    # calculate lpips between consecutive frames in each video, then take average.
    # now I don't know if I should compare this to the GT video or not, since the other metrics are already computing loss between SR and GT frames
    total_lpips = 0
    total_frames = len(glob(video1_dir + '/*.jpg'))  # should be the same for both videos, TODO: check for consistency
    
    loss_fn = lpips.LPIPS(net='alex') 
    
    frames = sorted(glob(video_dir + '/*.jpg')) 
    for i, img_path in enumerate(frames):
        img1 = cv2.imread(img_path)
        img2 = cv2.imread(frames[i + 1]) if i + 1 < len(frames) else img1  # wrap around to the first frame
    pass