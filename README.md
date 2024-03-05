# NT-ViT: Neural Transcoding Vision Transformers for EEG-to-fMRI Synthesis

This is a PyTorch implementation of NT-ViT proposed in our paper "Neural Transcoding Vision Transformers for EEG-to-fMRI Synthesis".



## Environment setup
- Install [Anaconda](https://www.anaconda.com/download)

- Create a new `conda` environment:
    ```bash
    conda create --name ntvit python=3.9 && conda activate ntvit
    ```
- Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
And that's it, ready to go!

## Dataset preprocessing
In order to be used with the model, the datasets must be preprocessed to extract the EEGs and fMRIs
In order to do so:
- download the datasets
    - NODDI dataset can be downloaded from [here](https://osf.io/94c5t/), and its structure is this one:
    ```
    |-- EEG1
    |   |-- 32
    |   |   |-- export
    |   |   |   |-- ...
    |   |   `-- raw
    |   |       |-- ...
    |   |-- ...
    |   |  
    |-- EEG2
    |   |-- 43
    |   |   |-- export
    |   |   |   |-- ...
    |   |   `-- raw
    |   |       |-- ...
    |   |-- ...
    `-- fMRI
        |-- 32
        |   |-- ...
        |-- ...
    ```
    - Oddball dataset can be downloaded from [here](https://openneuro.org/datasets/ds000116/versions/00003), and its structure is this:
    ```
    |-- sub001
    |   |-- BOLD
    |   |   |-- ..
    |   |-- EEG
    |   |   |-- ..
    |   |-- ...
    |   |  
    |-- sub002
    |   |-- ...
    |-- ...
    ```
- preprocess the datasets by running the following:
    ```bash 
    python preprocess_datasets.py --dataset_type=noddi --dataset_path=PATH_TO_DATASET --output_path=PATH_TO_PREPROCESSED_DATASET;
    python preprocess_datasets.py --dataset_type=oddball --dataset_path=PATH_TO_DATASET --output_path=PATH_TO_PREPROCESSED_DATASET
    ```
For the sake of the review, a portion of the preprocessed datasets containing a sample from each subject is attached to this code, in the `datasets_preprocessed` folder. 

## Training a model
To train a model using the LOSO scheme, simply run `train.py`. 
The logs will be saved to [wandb](wandb.ai), so you will be prompted to login on the first time running the script.

For example, this is the command to train a model on the NODDI dataset:
```bash
python train.py --dataset_type=noddi --dataset_path=PATH_TO_PREPROCESSED_DATASET --use_domain_matching
```
To train a model on the Oddball dataset, `--dataset_type` must be given `oddball` as parameter:
```bash
python train.py --dataset_type=oddball --dataset_path=PATH_TO_PREPROCESSED_DATASET --use_domain_matching
```

The description of the parameters accepted by the script can be viewed using `python train.py -h`.

## Pre-trained weights
Loading a pre-trained model given its `.ckpt` checkpoint can be done using the following command:
```python
model = NTViT.load_from_checkpoint(PATH_TO_WEIGHTS, map_location=torch.device('cpu'))
```
Due to limitations in file size for the supplementary material, no pre-trained weights can be attached to this code.
However, they will be distributed after the final decision.

## Comparison with SOTA
### NODDI dataset
| Method              | RMSE $\downarrow$ | SSIM $\uparrow$ | PSNR $\uparrow$ |
|---------------------|-------------------|-----------------|-----------------|
| LP [1]              | 0.51              | 0.433           | -               |
| TALP [1]            | 0.41              | 0.472           | -               |
| FP [1]              | 0.43              | 0.462           | -               |
| TAFP [1]            | 0.40              | 0.462           | -               |
| CNN [2]             | 0.46              | 0.449           | -               |
| NT-ViT w/o DM, ours | 0.11              | 0.516           | 19.56           |
| NT-ViT, ours        | **0.10**              | **0.534**           | **20.05**           |

### Oddball dataset
| Method              | RMSE $\downarrow$ | SSIM $\uparrow$ | PSNR $\uparrow$ |
|---------------------|-------------------|-----------------|-----------------|
| LP [1]              | 0.74              | 0.183           | -               |
| TALP [1]            | 0.77              | 0.158           | -               |
| FP [1]              | 0.73              | 0.196           | -               |
| TAFP [1]            | 0.70              | 0.200           | -               |
| CNN [2]             | 0.86              | 0.189           | -               |
| NT-ViT w/o DM, ours | **0.07**              | 0.595           | 22.98           |
| NT-ViT, ours        | **0.07**              | **0.613**           | **23.09**           |

## References
1) Calhas, D., Henriques, R.: Eeg to fmri synthesis benefits from attentional graphs of electrode relationships (2022). https://arxiv.org/abs/2203.03481
2) Liu, X., Sajda, P.: A convolutional neural network for transcoding simultaneously550 550
acquired eeg-fmri data. In: 2019 9th International IEEE/EMBS Conference on551 551
Neural Engineering (NER). pp. 477–482 (2019). https://doi.org/10.1109/NER.2019.8716994