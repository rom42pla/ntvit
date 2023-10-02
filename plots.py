from importlib import reload
from os import makedirs, pardir, remove
from os.path import join, isdir, abspath
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
        nrows=n_channels,
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
    fig.text(0.05,0.5,"Channels", rotation="vertical")
        
    # normalizes the images
    if not vmin:
        vmin = min(image.get_array().min() for image in images)
    if not vmax:
        vmax = max(image.get_array().max() for image in images)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for image in images:
        image.set_norm(norm)

    fig.colorbar(images[0], ax=axs, orientation='vertical', fraction=0.025, label="Amplitude")

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

if __name__ == "__main__":
    plot_reconstructed_ecgs_waveforms(
        ecgs_pred_wf=torch.randn(2, 128),
        ecgs_gt_wf=torch.randn(2, 128),
        path="./tmp_wf.png"
    )
    plot_reconstructed_spectrograms(
        sg_pred=torch.randn(14, 128, 8),
        sg_gt=torch.randn(14, 128, 8),
        path="./tmp_sg.png"
    )