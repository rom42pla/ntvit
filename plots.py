from importlib import reload
from os import makedirs, pardir, remove
from os.path import join, isdir, abspath
import numpy as np
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datasets.eeg_fmri_preprocessed import NoddiPreprocessedDataset
from utils import set_seed

matplotlib.use("agg")
matplotlib = reload(matplotlib)
import torch


def create_parent_directory(path):
    parent_directory = abspath(join(path, pardir))
    if not isdir(parent_directory):
        makedirs(parent_directory)


def show_image():
    try:
        plt.show()
    except Exception as e:
        print("failed to show the figure")


def read_image(path):
    return matplotlib.image.imread(path)


def save_image(path):
    plt.savefig(path)


def plot_reconstructed_ecgs_waveforms(
    ecgs_pred_wf, ecgs_gt_wf, scale=4, dpi=200, path=None, show=False
):
    assert (
        ecgs_pred_wf.shape == ecgs_gt_wf.shape
    ), f"ecgs_pred shape ({tuple(ecgs_pred_wf.shape)}) != ecgs_gt shape ({tuple(ecgs_gt_wf.shape)})"
    assert (
        len(ecgs_pred_wf.shape) == 2
    ), f"shape must be (channels, time), got {tuple(ecgs_pred_wf.shape)}"
    if isinstance(ecgs_pred_wf, torch.Tensor):
        ecgs_pred_wf = ecgs_pred_wf.detach().cpu()
    if isinstance(ecgs_gt_wf, torch.Tensor):
        ecgs_gt_wf = ecgs_gt_wf.detach().cpu()
    n_channels = ecgs_pred_wf.shape[0]

    # builds the layout of the figure
    fig, axs = plt.subplots(
        nrows=n_channels,
        ncols=2,
        dpi=dpi,
        sharex=True,
        sharey=True,
        tight_layout=True,
        figsize=(scale, scale),
    )
    axs[0, 0].set_title("Ground truth")
    axs[0, 1].set_title("Generated")

    # plots the ecgs
    for i_channel in range(n_channels):
        axs[i_channel, 0].set_ylabel(f"Channel {i_channel}")
        axs[i_channel, 0].plot(ecgs_gt_wf[i_channel])
        axs[i_channel, 1].plot(ecgs_pred_wf[i_channel])

    # shows the figure
    if show:
        show_image()

    # saves the figure
    if path:
        create_parent_directory(path)
        tmp_filepath = path
    else:
        tmp_filepath = ".tmp.png"
    save_image(tmp_filepath)
    plt.close(fig)
    image = read_image(tmp_filepath)

    # eventually remove the generated temporary file
    if not path:
        remove(tmp_filepath)
    return image


def plot_reconstructed_spectrograms(
    sg_pred,
    sg_gt,
    vmin=None,
    vmax=None,
    scale=4,
    dpi=200,
    cmap="hot",
    path=None,
    show=False,
):
    assert (
        sg_pred.shape == sg_gt.shape
    ), f"ecgs_pred_sg shape ({tuple(sg_pred.shape)}) != ecgs_gt_sg shape ({tuple(sg_gt.shape)})"
    assert (
        len(sg_pred.shape) == 3
    ), f"shape must be (channels, frequencies, time), got {tuple(sg_pred.shape)}"
    if isinstance(sg_pred, torch.Tensor):
        sg_pred = sg_pred.detach().cpu()
    if isinstance(sg_gt, torch.Tensor):
        sg_gt = sg_gt.detach().cpu()
    n_channels = sg_pred.shape[0]

    # builds the layout of the figure
    fig, axs = plt.subplots(
        nrows=max(2, n_channels),
        ncols=2,
        dpi=dpi,
        sharex=True,
        sharey=True,
    )
    axs[0, 0].set_title("Real data")
    axs[0, 1].set_title("Generated data")

    # plots the ecgs
    images = []
    for i_channel in range(n_channels):
        # axs[i_channel, 0].set_ylabel(f"Channel {i_channel}")
        images.extend(
            (
                axs[i_channel, 0].imshow(
                    sg_gt[i_channel], vmin=vmin, vmax=vmax, cmap=cmap
                ),
                axs[i_channel, 1].imshow(
                    sg_pred[i_channel], vmin=vmin, vmax=vmax, cmap=cmap
                ),
            )
        )
    for ax in axs.flat:
        ax.set_yticks([])
        ax.set_aspect("auto")
    fig.text(0.05, 0.5, "Channels", rotation="vertical")

    # normalizes the images
    if not vmin:
        vmin = min(image.get_array().min() for image in images)
    if not vmax:
        vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for image in images:
        image.set_norm(norm)

    fig.colorbar(
        images[0], ax=axs, orientation="vertical", fraction=0.025, label="Amplitude"
    )

    # shows the figure
    if show:
        show_image()

    # saves the figure
    if path:
        create_parent_directory(path)
        tmp_filepath = path
    else:
        tmp_filepath = ".tmp.png"
    save_image(tmp_filepath)
    plt.close(fig)
    image = read_image(tmp_filepath)

    # eventually remove the generated temporary file
    if not path:
        remove(tmp_filepath)
    return image


def plot_reconstructed_fmris(
    fmris_pred,
    fmris_gt,
    vmin=None,
    vmax=None,
    scale=4,
    dpi=200,
    cmap="gray",
    path=None,
    show=False,
):
    assert (
        fmris_pred.shape == fmris_gt.shape
    ), f"fmris_pred shape ({tuple(fmris_pred.shape)}) != fmris_gt shape ({tuple(fmris_gt.shape)})"
    assert (
        len(fmris_pred.shape) == 3
    ), f"shape must be (y, x, z), got {tuple(fmris_pred.shape)}"
    if isinstance(fmris_pred, torch.Tensor):
        fmris_pred = fmris_pred.detach().cpu()
    if isinstance(fmris_gt, torch.Tensor):
        fmris_gt = fmris_gt.detach().cpu()
    n_channels = fmris_pred.shape[0]

    # builds the layout of the figure
    fig, axs = plt.subplots(
        nrows=n_channels,
        ncols=3,
        dpi=dpi,
        sharex=True,
        sharey=True,
    )
    axs[0, 0].set_title("Real data")
    axs[0, 1].set_title("Generated data")
    axs[0, 2].set_title("Difference")

    # plots the ecgs
    images = []
    for i_channel in range(n_channels):
        # axs[i_channel, 0].set_ylabel(f"Channel {i_channel}")
        images.extend(
            (
                axs[i_channel, 0].imshow(
                    fmris_gt[i_channel], vmin=vmin, vmax=vmax, cmap=cmap
                ),
                axs[i_channel, 1].imshow(
                    fmris_pred[i_channel], vmin=vmin, vmax=vmax, cmap=cmap
                ),
            )
        )
        axs[i_channel, 2].imshow(
           fmris_gt[i_channel] - fmris_pred[i_channel],
            cmap=cmap,
        )
    for ax in axs.flat:
        ax.set_yticks([])
        ax.set_aspect("auto")
    fig.text(0.05, 0.5, "Channels", rotation="vertical")

    # normalizes the images
    if not vmin:
        vmin = min(image.get_array().min() for image in images)
    if not vmax:
        vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for image in images:
        image.set_norm(norm)

    fig.colorbar(
        images[0], ax=axs, orientation="vertical", fraction=0.025, label="Intensity"
    )

    # shows the figure
    if show:
        show_image()

    # saves the figure
    if path:
        create_parent_directory(path)
        tmp_filepath = path
    else:
        tmp_filepath = ".tmp.png"
    save_image(tmp_filepath)
    plt.close(fig)
    image = read_image(tmp_filepath)

    # eventually remove the generated temporary file
    if not path:
        remove(tmp_filepath)
    return image


def plot_reconstructed_ecgs_and_eegs(eeg, ecg_gt, ecg_pred, scale=4, path=None):
    assert len(eeg.shape) == 2, f"shape must be (channels, time), got {eeg.shape}"
    assert len(ecg_gt.shape) == 2, f"shape must be (channels, time), got {ecg_gt.shape}"
    assert (
        len(ecg_pred.shape) == 2
    ), f"shape must be (channels, time), got {ecg_pred.shape}"
    if isinstance(eeg, torch.Tensor):
        eeg = eeg.detach().cpu()
    if isinstance(ecg_gt, torch.Tensor):
        ecg_gt = ecg_gt.detach().cpu()
    if isinstance(ecg_pred, torch.Tensor):
        ecg_pred = ecg_pred.detach().cpu()
    fig = plt.figure(tight_layout=True, figsize=(3 * scale, scale))
    gs = gridspec.GridSpec(4, 4 * 3)

    axs_eeg = []
    for x in range(4):
        for y in range(4):
            if len(axs_eeg) == eeg.shape[0]:
                break
            ax = fig.add_subplot(gs[y, x])
            axs_eeg.append(ax)
            ax.plot(eeg[len(axs_eeg) - 1])
        if len(axs_eeg) == eeg.shape[0]:
            break
    ax_title_eeg = fig.add_subplot(gs[:4])
    ax_title_eeg.axis("off")
    ax_title_eeg.set_title("EEGs")

    axs_ecg_gt = []
    for y in range(2):
        ax = fig.add_subplot(gs[y * 2 : y * 2 + 2, 4 : 4 * 2])
        axs_ecg_gt.append(ax)
        ax.plot(ecg_gt[len(axs_ecg_gt) - 1])
    ax_title_ecg_gt = fig.add_subplot(gs[4 : 4 * 2])
    ax_title_ecg_gt.axis("off")
    ax_title_ecg_gt.set_title("ECGs GT")

    axs_ecg_pred = []
    for y in range(2):
        ax = fig.add_subplot(gs[y * 2 : y * 2 + 2, 4 * 2 : 4 * 3])
        axs_ecg_pred.append(ax)
        ax.plot(ecg_pred[len(axs_ecg_pred) - 1])
    ax_title_ecg_pred = fig.add_subplot(gs[4 * 2 : 4 * 3])
    ax_title_ecg_pred.axis("off")
    ax_title_ecg_pred.set_title("ECGs pred")

    gs.tight_layout(fig)
    filepath = ".tmp.png"
    if not path:
        plt.show()
    else:
        parent_directory = abspath(join(path, pardir))
        if not isdir(parent_directory):
            makedirs(parent_directory)
            print(f"created {parent_directory}")
        filepath = path
    fig.savefig(filepath)
    image = read_image(filepath)
    if not path:
        remove(path)
    return image

def plot_reconstructed_fmris(
    fmris_pred,
    fmris_gt,
    vmin=None,
    vmax=None,
    scale=4,
    dpi=200,
    mode: str = "pc",
    cmap="coolwarm",
    path=None,
    show=False,
):
    assert (
        fmris_pred.shape == fmris_gt.shape
    ), f"ecgs_pred_sg shape ({tuple(fmris_pred.shape)}) != ecgs_gt_sg shape ({tuple(fmris_gt.shape)})"
    assert (
        len(fmris_pred.shape) == 3
    ), f"shape must be (z, y, x), got {tuple(fmris_pred.shape)}"
    if isinstance(fmris_pred, torch.Tensor):
        fmris_pred = fmris_pred.detach().cpu().numpy()
    if isinstance(fmris_gt, torch.Tensor):
        fmris_gt = fmris_gt.detach().cpu().numpy()
    assert mode in {"pc", "mip"}

    max_value = max(fmris_gt.max(), fmris_pred.max())
    fmris_pred = fmris_pred / max_value
    fmris_gt = fmris_gt / max_value
    
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
    
    threshold = threshold_otsu(np.max(fmris_gt, axis=0))

    fig = plt.figure(figsize=(scale * 2, scale), dpi=300, tight_layout=True)
    # ax_pred = fig.add_subplot(121, projection='3d')
    # ax_gt = fig.add_subplot(122, projection='3d')
    # Define a gridspec with two rows and two columns
    gs = gridspec.GridSpec(3, 2, height_ratios=[1, 0.05, 0.05])
    if mode in {"pc"}:
        ax_gt = fig.add_subplot(gs[0, 0], projection='3d')
        ax_pred = fig.add_subplot(gs[0, 1], projection='3d')
    else:
        ax_gt = fig.add_subplot(gs[0, 0])
        ax_pred = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[2, :])
    
    # Generate meshgrid for coordinates
    z, y, x = np.meshgrid(np.arange(fmris_pred.shape[0]), np.arange(fmris_pred.shape[1]), np.arange(fmris_pred.shape[2]), indexing='ij')

    # Flatten the meshgrids to get xs, ys, zs
    xs, ys, zs = x.flatten(), y.flatten(), z.flatten()
    
    def plot_pointcloud(fmri, ax):
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

        ax.set_xlabel('X')
        ax.set_xlim(0, fmris_gt.shape[2])
        ax.set_ylabel('Y')
        ax.set_ylim(0, fmris_gt.shape[1])
        ax.set_zlabel('Z')
        ax.set_zlim(0, fmris_gt.shape[0])
        return im
    
    def plot_mip(fmri, ax):
        mip = np.max(fmri, 0)  # Maximum Intensity Projection along z-axis
        im = ax.imshow(mip, cmap="coolwarm")
        
        ax.set_xlabel('X')
        ax.set_xlim(0, fmris_gt.shape[2])
        ax.set_ylabel('Y')
        ax.set_ylim(0, fmris_gt.shape[1])
        return im
        
    if mode == "pc":
        im = plot_pointcloud(fmri=fmris_pred, ax=ax_pred)
        plot_pointcloud(fmri=fmris_gt, ax=ax_gt)
    elif mode == "mip":
        im = plot_mip(fmri=fmris_pred, ax=ax_pred)
        plot_mip(fmri=fmris_gt, ax=ax_gt)
    
    # Create a single colorbar using the cbar_ax we defined earlier
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal', aspect=2**7)
    cbar.set_label('Intensity')
    
    ax_gt.set_title(f"Ground truth")
    ax_pred.set_title(f"Generated")

    # shows the figure
    if show:
        show_image()

    # saves the figure
    if path:
        create_parent_directory(path)
        tmp_filepath = path
    else:
        tmp_filepath = ".tmp.png"
    save_image(tmp_filepath)
    plt.close(fig)
    image = read_image(tmp_filepath)

    # eventually remove the generated temporary file
    if not path:
        remove(tmp_filepath)
    return image

def plot_fmris(fmris, save_path: str, mode="mip", scale=4):
    assert len(fmris.shape) == 3
    import matplotlib

    matplotlib.use("agg")
    import matplotlib.pyplot as plt

    if mode == "mip":
        fig, ax_pc = plt.subplots(1, 1, figsize=(scale, scale), dpi=300)
        mip = np.max(fmris, axis=0)  # Maximum Intensity Projection along z-axis
        im = ax_pc.imshow(mip, cmap="gray")

        # colorbar
        cbar_ax = fig.add_axes(
            [0.85, 0.15, 0.01, 0.65]
        )  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        fig.subplots_adjust(right=0.8)  # make room for colorbar
        fig.suptitle(f"Maximum Intensity Projection of fMRI")
        fig.savefig(save_path)
        fig.clf()
    elif mode == "pc":
        fmris = fmris / fmris.max()

        # Generate meshgrid for coordinates
        z, y, x = np.meshgrid(np.arange(fmris.shape[0]), np.arange(fmris.shape[1]), np.arange(fmris.shape[2]), indexing='ij')

        # Flatten the meshgrids to get xs, ys, zs
        xs = x.flatten()
        ys = y.flatten()
        zs = z.flatten()

        # Flatten tensor for coloring and transparency
        t_low = fmris.flatten()

        # 1. Dimensionality and Clutter: Apply threshold         
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
        
        # threshold = np.percentile(fmris, q=90)
        threshold = threshold_otsu(np.max(fmris, axis=0))
        mask = t_low > threshold

        # 2. Transparency: Adjust transparency mapping (using a simple linear scaling in this example)
        alpha_values = t_low/t_low.max()
        # alpha_values = alpha_values * 0.8 + 0.2  # This ensures values are in the range [0.2, 1]

        # 3. Color: Use a diverging colormap
        colormap = plt.cm.coolwarm

        # 4. Size: Map intensity to point sizes
        point_sizes = (t_low * 50) + 10  # Adjust scaling factor and base size as needed

        xs = xs[mask]
        ys = ys[mask]
        zs = zs[mask]
        t_low = t_low[mask]
        alpha_values = alpha_values[mask]
        point_sizes = point_sizes[mask]

        fig = plt.figure(figsize=(scale, scale*1.25), dpi=300, tight_layout=True)
        ax_pc = fig.add_subplot(projection='3d')
        
        im = ax_pc.scatter(xs, ys, zs, c=t_low, cmap=colormap, alpha=alpha_values, s=point_sizes, vmin=0, vmax=1)
        cbar = fig.colorbar(im, ax=ax_pc, orientation='horizontal', pad=0.1, aspect=2**6)
        cbar.set_label('Intensity')

        ax_pc.set_xlabel('X')
        ax_pc.set_ylabel('Y')
        ax_pc.set_zlabel('Z')

        fig.savefig(save_path)
        fig.clf()
            
if __name__ == "__main__":
    set_seed(42)
    dataset_path = f"../../datasets/eeg_fmri_prep_512hz"
    dataset = NoddiPreprocessedDataset(
        path=dataset_path,
        normalize_eegs=True,
        normalize_fmris=True,
    )
    eegs = dataset[0]["eegs"]
    fmris = dataset[0]["fmris"]
    
    # plot_reconstructed_ecgs_waveforms(
    #     np.random.randn(*eegs.shape),
    #     eegs,
    #     path="./tmp_wf.png",
    # )
    # plot_reconstructed_spectrograms(
    #     np.random.randn(*eegs.shape),
    #     eegs,
    #     path="./tmp_sg.png",
    # )
    plot_reconstructed_fmris(
        fmris_pred=np.random.rand(*fmris.shape),
        fmris_gt=fmris,
        mode="pc",
        path="./tmp_fmris.png",
    )
