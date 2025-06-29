# VSR-CSE244c

A comparison between SOTA Video Super Resolution (VSR) models.
Writeup can be found [here](https://github.com/dakshshah03/VSR-CSE244c/blob/main/CSE_244c_Final_Project_Report-4.pdf)
We will be comparing the following models:
- Bicubic Interpolation (baseline)
- ESRGAN (GAN)
- Real-ESRGAN (GAN)
- SR3 (Diffusion)
- StableVSR (Diffusion w/ Temporal Consistency)

## Getting Started
1. Clone the repository
    ```
    git clone https://github.com/dakshshah03/VSR-CSE244c.git
    ```
2. Initialize submodules
   ```
   cd VSR-CSE244c
   git submodule update --init
   git submodule update --remote
   ```
### Setting up the Benchmark Dataset
1. Download the `train_sharp` and `train_sharp_bicubic` datasets from https://seungjunnah.github.io/Datasets/reds.html
2. Extract the sequences 000, 011, 015, 020 from both low resolution and ground truth, organized as:
```
data
└── REDS
    └── benchmark_data
        ├── ground_truth
        │   ├── 000
        │   ├── 011
        │   ├── 015
        │   └── 020
        └── low_resolution
            ├── 000
            ├── 011
            ├── 015
            └── 020
```
### Evaluation Metrics
Evaluates the outputs of VSR models using "SSIM", "PSNR", "DISTS", "LPIPS", and "tLPIPS"
(We have put aside a saved table of a run through of all the metrics so you don't have to yourself at `saved_metrics/model_metrics.csv`)

#### Requirements
- torch (https://pytorch.org/get-started/locally/)
- pandas
- pillow
- torchvision
- torchmetrics

1. Run
    ```
    pip install pandas torch torchvision torchmetrics
    ```
2. Download the groundtruth dataset and VSR predicions
3. Organize the REDS4 input data as detailed in the setup instructions earlier
4. Organize the outputs as:
    ```
    output
    ├── baseline
    │   ├── 000
    │   ├── 011
    │   ├── 015
    │   └── 020
    ├── esrgan
    │   ├── 000
    │   ├── 011
    │   ├── 015
    │   └── 020
    ├── realesr
    │   ├── 000
    │   ├── 011
    │   ├── 015
    │   └── 020
    └── stablevsr
        ├── 000
        ├── 011
        ├── 015
        └── 020
    ```
5. run `python 'utils/evaluate_metrics.py' '<model1_predictions>' '<model2_predictions>' ...`
    - i.e. python 'utils/evaluate_metrics.py' 'baseline' 'esrgan' 'realesr' 'stablevsr'
6. read the output that is printed out to get metrics, it is also saved in `model_metrics.csv`


## Bicubic Interpolation (Baseline)
### Requirements
- python 3.8+
- opencv
- tqdm

### Running
```
cd scripts && bash bicubic_inference.sh
```

which will upscale the sets of images from REDS4 in the `./data/REDS/benchmark_data/low_resolution` folder and output the upscaled images (and videos) in `./output/baseline/`

## ESRGAN
### Setting up
1. Run
   ```
   conda create -n esrgan python=3.8 -y
   conda activate esrgan
   pip install numpy opencv-python
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
2. Download the model weights and place them in the `./submodules/ESRGAN/models/` directory from their [google drive]([url](https://drive.google.com/drive/u/0/folders/17VYV_SoZZesU6mbxz2dMAIccSSlqLecY)).
### Running Inference
```
cd scripts && bash esrgan_inference.sh
```

which will upscale the sets of images from REDS4 in the `./data/REDS/benchmark_data/low_resolution` folder and output the upscaled images (and videos) in `./output/esrgan/`

## Real-ESRGAN
### Setting up:
1. Make sure CUDA 11.8 is installed (unsure if this is required but to be safe)
2. Create the conda environment:
    ```
    conda create -n "real-esrgan" python=3.8
    ```
3. Activate the environment:
    ```
    conda activate real-esrgan
    ```
4. Install required packages:
    ```
    pip install basicsr
    pip install facexlib
    pip install gfpgan
    cd submodules/Real-ESRGAN
    pip install -r requirements.txt
    python setup.py develop
    ```
5. Since BasicSR uses an older version of pytorch, we need to modify one import in its folder.
This fix is based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/13985#issuecomment-1813885266 
    1. Navigate to this directory: `/home/ubuntu/miniconda3/envs/real-esrgan/lib/python3.8/site-packages/basicsr/data/`
    2. Open `degradations.py`
    3. Change the import `from torchvision.transforms.functional_tensor import rgb_to_grayscale` \
     to `from torchvision.transforms.functional import rgb_to_grayscale`

### Running Inference
We have set up a script to run inference on REDS4.
    ```
    cd scripts && bash realesrgan_inference.sh
    ```
which will upscale the images in the `./data/REDS/benchmark_data/low_resolution` folder and output the upscaled images (and videos) in `./output/realesr/`

## StableVSR
### Setting up
1. Run 
    ```
    conda create -n stablevsr python=3.8.17 -y
    cd submodules/StableVSR
    conda activate stablevsr
    pip install -r requirements.txt
    ```

### Running Inference
We have set up a script to run inference on REDS4.
    ```
    cd scripts && bash stablevsr_inference.sh
    ```
which will upscale the images in the `./data/REDS/benchmark_data/low_resolution` folder and output the upscaled images (and videos) in `./output/stablevsr/`

<!-- ## Baseline
Bicubic Interpolation will be used to upscale the images.

Requirements:
```
- python >= 3.8
- opencv
```

## Evaluation Metrics
### SSIM

```
Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). Image quality assessment: From error visibility to structural similarity. IEEE Transactions on Image Processing, 13, 600-612. https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf, DOI:10.1109/TIP.2003.819861
```

### LPIPS
https://github.com/richzhang/PerceptualSimilarity 

### 

### Requirements
```
- skimage
- lpips
- opencv-python
- 
``` -->

