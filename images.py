from os import makedirs
from os.path import join
import random
from einops import rearrange
import numpy as np
from math import floor, ceil
import torch
from torch import nn
from torch.nn import functional as F
import torchaudio
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
matplotlib.use("agg")

def read_image(path):
    return matplotlib.image.imread(path)


def save_image(path):
    plt.savefig(path, bbox_inches='tight')
    
def threshold_otsu(gray_img, nbins=.1):
    all_pixels = gray_img.flatten()
    p_all = len(all_pixels)
    least_variance = -1
    least_variance_threshold = -1
    
    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(gray_img)+nbins, np.max(gray_img)-nbins, nbins)
    
    # loop through the thresholds to find the one with the least class variance
    for color_threshold in color_thresholds:
        # background
        bg_pixels = all_pixels[all_pixels < color_threshold]
        p_bg = len(bg_pixels)
        w_bg = p_bg / p_all
        variance_bg = np.var(bg_pixels)
        
        # foreground
        fg_pixels = all_pixels[all_pixels >= color_threshold]
        p_fg = len(fg_pixels)
        w_fg = p_fg / p_all
        variance_fg = np.var(fg_pixels)
        
        variance = w_bg * variance_bg + w_fg * variance_fg
        
        if least_variance == -1 or variance < least_variance:
            least_variance = variance
            least_variance_threshold = color_threshold
    return least_variance_threshold
    
def get_patches(input, patches_size, padding):
    padded_input = F.pad(input, pad=padding)
    patches = padded_input.unfold(2, patches_size, patches_size).unfold(3, patches_size, patches_size)
    patches = rearrange(patches, "b c p1 p2 h w -> b (p1 p2) c h w")
    return patches
    
def plot_pointcloud(fmri, ax, threshold, add_ticks=True):
    z, y, x = np.meshgrid(np.arange(fmri.shape[0]), np.arange(fmri.shape[1]), np.arange(fmri.shape[2]), indexing='ij')
    xs, ys, zs = x.flatten(), y.flatten(), z.flatten()
    
    # Flatten tensor for coloring and transparency
    t_low = fmri.flatten()

    # 1. Dimensionality and Clutter: Apply threshold 
    # threshold = np.percentile(fmri, q=75)

    mask = t_low > threshold

    # 2. Transparency: Adjust transparency mapping (using a simple linear scaling in this example)
    alpha_values = t_low
    # alpha_values = alpha_values * 0.8 + 0.2  # This ensures values are in the range [0.2, 1]

    # 4. Size: Map intensity to point sizes
    point_sizes = (t_low * 10) + 1  # Adjust scaling factor and base size as needed
    mask = mask if mask.sum() > 0 else t_low > np.percentile(fmri, 75)
    
    im = ax.scatter(xs[mask], ys[mask], zs[mask], c=t_low[mask], cmap="coolwarm", alpha=alpha_values[mask], s=point_sizes[mask], vmin=0, vmax=1)
    # The fix
    for spine in ax.spines.values():
        spine.set_visible(False)
    if add_ticks:
        ax.set_xlabel('X')
        ax.set_xlim(0, fmri.shape[2])
        ax.set_ylabel('Y')
        ax.set_ylim(0, fmri.shape[1])
        ax.set_zlabel('Z')
        ax.set_zlim(0, fmri.shape[0])
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    return im

def plot_mip(fmri, ax, add_ticks=True):
    mip = fmri.max(0)
    im = ax.imshow(mip, cmap="coolwarm", vmin=0, vmax=1)
    if add_ticks:
        ax.set_xlabel('X')
        ax.set_xlim(0, fmri.shape[0])
        ax.set_ylabel('Y')
        ax.set_ylim(0, fmri.shape[1])
    else:
        plt.axis('off')
    return im

def plot_waveform(waveform, ax, add_ticks=True):
    assert len(waveform.shape) == 1
    
    im = ax.plot(np.arange(len(waveform)), waveform)
    if add_ticks:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return im

def plot_spectrogram(spectrogram, ax, add_ticks=True):
    assert len(spectrogram.shape) == 2
    
    im = ax.imshow(spectrogram, aspect="auto", cmap="coolwarm", origin="lower")
    if add_ticks:
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    return im
    
def get_padding_for_patches(height, width, kernel_size):
    padding_w = kernel_size * ceil(width / kernel_size) - width
    padding_h = kernel_size * ceil(height / kernel_size) - height
    if padding_w % 2 == 0:
        padding_left = padding_right = padding_w // 2
    else:
        padding_left = floor(padding_w / 2)
        padding_right = padding_left + 1
    if padding_h % 2 == 0:
        padding_up = padding_down = padding_h // 2
    else:
        padding_up = floor(padding_h / 2)
        padding_down = padding_up + 1
    return padding_left, padding_right, padding_up, padding_down
    
def plot_single_waveform(waveform, path, add_ticks: bool = False, scale=4, dpi=300):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()

    # builds the figure
    fig = plt.figure(figsize=(scale, scale), dpi=dpi, tight_layout=True)
    nrows = min(6, waveform.shape[0])
    gs = gridspec.GridSpec(nrows=nrows, ncols=1)
    for i_channel in range(nrows):
        ax = fig.add_subplot(gs[i_channel, :])
        if i_channel == nrows -2:
            ax.set_axis_off()
            ax.text(0.5, 0.75, "...", ha='center', va='center', fontsize=32) 
            continue
        plot_waveform(waveform=waveform[i_channel], ax=ax, add_ticks=add_ticks)

    # saves the figure
    save_image(path)
    plt.close(fig)
    image = read_image(path)
    return image

def plot_single_spectrogram(spectrogram, path, add_ticks: bool = False, scale=4, dpi=300):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu().numpy()

    # builds the figure
    fig = plt.figure(figsize=(scale, scale), dpi=dpi)
    nrows = min(6, spectrogram.shape[0])
    gs = gridspec.GridSpec(nrows=nrows, ncols=1)
    for i_channel in range(nrows):
        ax = fig.add_subplot(gs[i_channel, :])
        if i_channel == nrows -2:
            ax.set_axis_off()
            ax.text(0.5, 0.75, "...", ha='center', va='center', fontsize=32) 
            continue
        plot_spectrogram(spectrogram=spectrogram[i_channel], ax=ax, add_ticks=add_ticks)

    # saves the figure
    save_image(path)
    plt.close(fig)
    image = read_image(path)
    return image

def plot_spectrogram_patches(spectrogram, path, add_ticks: bool = False, patches_size=8, n_patches=8, scale=4, dpi=300):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.detach().cpu()
    spectrogram = spectrogram / spectrogram.max()

    # builds the figure
    fig = plt.figure(figsize=(n_patches, 1), dpi=dpi)
    gs = gridspec.GridSpec(nrows=1, ncols=n_patches)
    
    # splits the spectrogram into patches
    padding = get_padding_for_patches(
        width=spectrogram.shape[-1],
        height=spectrogram.shape[-2],
        kernel_size=patches_size,
    )    
    patches = get_patches(spectrogram.unsqueeze(0), patches_size=patches_size, padding=padding)[0]
    i_patch = -1
    for x in range(n_patches):
        i_patch += 1
        ax = fig.add_subplot(gs[x])
        ax.set_axis_off()
        if i_patch == n_patches-2:
            ax.text(0.5, 0.6, "...", ha='center', va='center', fontsize=32) 
            continue
        plot_spectrogram(spectrogram=patches[i_patch, 0, ...], ax=ax, add_ticks=add_ticks)

    # saves the figure
    save_image(path)
    plt.close(fig)
    image = read_image(path)
    return image

def plot_fmri_patches(fmri, path, add_ticks: bool = False, patches_size=8, n_patches=8, scale=4, dpi=300):
    if isinstance(fmri, torch.Tensor):
        fmri = fmri.detach().cpu()
    fmri = fmri / fmri.max()

    # builds the figure
    fig = plt.figure(figsize=(n_patches, 1), dpi=dpi, tight_layout=True)
    gs = gridspec.GridSpec(nrows=1, ncols=n_patches)
    
    # splits the spectrogram into patches
    padding = get_padding_for_patches(
        width=fmri.shape[-1],
        height=fmri.shape[-2],
        kernel_size=patches_size,
    )    
    patches = get_patches(fmri.unsqueeze(0), patches_size=patches_size, padding=padding)[0]
    threshold = threshold_otsu(patches.amax(0).numpy())
    i_patch = -1
    for x in range(n_patches):
        i_patch += 1
        ax = fig.add_subplot(gs[x], projection="3d")
        ax.set_axis_off()
        if i_patch == n_patches-2:
            # ax.set_axis_off()
            ax.text(0.5, 0.6, 0.5, "...", ha='center', va='center', fontsize=32) 
            continue
        plot_pointcloud(fmri=patches[patches.shape[0]//2+i_patch], ax=ax, add_ticks=add_ticks, threshold=threshold)

    # saves the figure
    save_image(path)
    plt.close(fig)
    image = read_image(path)
    return image

def plot_single_fmri(fmri, path, mode="pc", add_colorbar: bool = True, add_ticks: bool = False, scale=4, dpi=300):
    assert mode in {"pc", "3d", "mip", "2d"}
    if isinstance(fmri, torch.Tensor):
        fmri = fmri.detach().cpu().numpy()
        
    assert fmri.min() >= 0
    if fmri.max() > 1:
        fmri = fmri / fmri.max()

    # builds the figure
    fig = plt.figure(figsize=(scale, scale), dpi=dpi, tight_layout=True)
    if add_colorbar:
        gs = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[1., 0.01])
    else:
        gs = gridspec.GridSpec(nrows=1, ncols=1)
    if mode in {"pc", "3d"}:
        ax = fig.add_subplot(gs[0, :], projection='3d')
        threshold = threshold_otsu(np.max(fmri, axis=0))
        im = plot_pointcloud(fmri=fmri, ax=ax, threshold=threshold, add_ticks=add_ticks)
    else:
        ax = fig.add_subplot(gs[0, :])
        im = plot_mip(fmri=fmri, ax=ax, add_ticks=add_ticks)
    # create a single colorbar using the cbar_ax we defined earlier
    if add_colorbar:
        cbar_ax = fig.add_subplot(gs[1, :])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', aspect=2**7)
        cbar.set_label('Intensity')

    # saves the figure
    save_image(path)
    plt.close(fig)
    image = read_image(path)
    return image



if __name__ == "__main__":
    # sets the random seed
    from utils import set_seed
    set_seed(42)
    
    # creates the output folder
    images_path = join("images")
    makedirs(images_path, exist_ok=True)
    
    # loads the dataset
    from datasets.eeg2fmri_preprocessed import EEG2fMRIPreprocessedDataset
    dataset_path = f"../../datasets/oddball_preprocessed"
    dataset = EEG2fMRIPreprocessedDataset(
        path=dataset_path,
        normalize_eegs=True,
        normalize_fmris=True,
    )
    # loads samples from the dataset
    sample_fmris = torch.stack(
        [
            torch.from_numpy(dataset[random.randint(0, len(dataset))]["fmris"])
            for _ in range(16)
        ], dim=0,
    )
    sample_eegs = torch.stack(
        [
            torch.from_numpy(dataset[random.randint(0, len(dataset))]["eegs"])
            for _ in range(16)
        ], dim=0,
    )
    # converts the eegs to spectrograms
    mel_spectrogrammer = torchaudio.transforms.MelSpectrogram(
            sample_rate=dataset.eegs_sampling_rate,
            n_fft=512,
            win_length=floor(dataset.eegs_sampling_rate*1/32),
            hop_length=floor(dataset.eegs_sampling_rate * 1/(32*2)),
            f_min=1,
            f_max=50,
            n_mels=16,
            power=2,
            normalized=False,
        )
    sample_mel_spectrograms = mel_spectrogrammer(sample_eegs.float())
    
    
    plot_single_fmri(fmri=sample_fmris[0], path=join(images_path, "fmri_no_ticks.png"), mode="pc", add_colorbar=False, add_ticks=False)
    plot_single_fmri(fmri=sample_fmris[0], path=join(images_path, "fmri_mip_no_ticks.png"), mode="mip", add_colorbar=False, add_ticks=False)
    plot_single_waveform(waveform=sample_eegs[0], path=join(images_path, "wf_no_ticks.png"), add_ticks=False)
    plot_single_spectrogram(spectrogram=sample_mel_spectrograms[0], path=join(images_path, "spec_no_ticks.png"), add_ticks=False)
    plot_spectrogram_patches(spectrogram=sample_mel_spectrograms[0], path=join(images_path, "spec_patches.png"), add_ticks=False)
    plot_fmri_patches(fmri=sample_fmris[0], path=join(images_path, "fmri_patches.png"), add_ticks=False)