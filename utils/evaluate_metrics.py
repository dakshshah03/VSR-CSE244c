import os
import sys
import metrics
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <pred_dir1> <pred_dir2> ...")
        sys.exit(1)

    data_path = os.path.join('data')
    ground_truth_path = os.path.join(data_path, 'benchmark_data', 'ground_truth')

    results = []

    # skip sys.argv[0] (script name)
    for arg in sys.argv[1:]:
        pred_path = os.path.join(data_path, arg)
        model_name = os.path.basename(pred_path.rstrip("/\\"))

        print(f"calculating {model_name}'s metrics...")

        avg_ssim, _ = metrics.ssim_video(ground_truth_path, pred_path)
        avg_psnr, _ = metrics.psnr_video(ground_truth_path, pred_path)
        avg_dists, _ = metrics.dists_video(ground_truth_path, pred_path)
        avg_lpips, _ = metrics.lpips_video(ground_truth_path, pred_path)
        avg_tlpips, _ = metrics.tlpips_video(pred_path)

        print(f"done alculating {model_name}'s metrics...\n")

        results.append({
            "Model": model_name,
            "SSIM": avg_ssim,
            "PSNR": avg_psnr,
            "DISTS": avg_dists,
            "LPIPS": avg_lpips,
            "tLPIPS": avg_tlpips,
        })

    # create DataFrame
    df = pd.DataFrame(results)
    df.set_index("Model", inplace=True)

    # print and save
    print(df)
    df.to_csv("model_metrics.csv")

if __name__ == "__main__":
    main()