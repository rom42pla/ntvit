from copy import deepcopy
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torchmetrics



matplotlib.use("agg")

class ImageAdjustment(torch.nn.Module):
    def __init__(self, image_size):
        super(ImageAdjustment, self).__init__()
        batch_size = image_size[0]
        self.offset = torch.nn.Parameter(
            torch.zeros(batch_size, 3), requires_grad=True
        )  # [batch_size, 3]
        self.image_size = image_size[1:]

    def forward(self, image):
        # Assuming image is of shape [batch_size, depth, height, width]
        batch_size = image.size(0)

        # Create a grid to apply the offset
        grid_z, grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, self.image_size[0]),
            torch.linspace(-1, 1, self.image_size[1]),
            torch.linspace(-1, 1, self.image_size[2]),
            indexing="ij",
        )
        grid = torch.stack([grid_z, grid_y, grid_x], dim=-1).to(
            image.device
        ).unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [batch_size, depth, height, width, 3]

        # Adjust grid for each image in the batch
        grid = grid + self.offset.view(
            batch_size, 1, 1, 1, 3
        )  # Broadcasting

        # Use grid_sample to move the image by the offset
        return F.grid_sample(
            image.unsqueeze(
                1
            ),  # Add channel dim: [batch_size, 1, depth, height, width]
            grid,
            align_corners=False,
            mode="bilinear",
        ).squeeze(
            1
        )  # Remove channel dim
            
def custom_loss(pred, gt, max_iterations=100, lr=1e-2, k=4, return_adjusted=False):
    adjuster = ImageAdjustment(image_size=pred.shape).to(pred.device)
    optimizer = torch.optim.AdamW(adjuster.parameters(), lr=lr)
    loss_min = torch.inf
    best_state_dict = adjuster.state_dict()

    pred_copied = pred.clone().detach()
    for i in range(max_iterations):
        with torch.enable_grad():
            optimizer.zero_grad()
            pred_adj = adjuster(pred_copied)
            loss = F.mse_loss(pred_adj, gt.detach())
            # loss = mse_patch_loss(adjusted_pred, gt.detach(), k=k)
            # if loss < loss_min:
            #     loss_min = loss
            #     best_state_dict = deepcopy(adjuster.state_dict())
            loss.backward()
            optimizer.step()

    # adjuster.load_state_dict(best_state_dict)
    pred_adj = adjuster(pred)
    loss = F.mse_loss(pred_adj, gt.detach())
    if return_adjusted:
        return loss, pred_adj
    else:
        return loss
    



def centroid_loss(pred, gt):
    def compute_centroids(volume):
        # Get the shape of the volume
        _, z, y, x = volume.shape

        # Create index tensors
        zz, yy, xx = torch.meshgrid(
            torch.linspace(0, 1, z, device=volume.device),
            torch.linspace(0, 1, y, device=volume.device),
            torch.linspace(0, 1, x, device=volume.device),
            indexing="ij",
        )
        # Compute the centroid
        total_voxels = (
            torch.sum(volume, dim=(1, 2, 3), keepdim=True) + 1e-10
        )  # Adding a small value to avoid division by zero
        z_centroid = torch.sum(zz * volume, dim=(1, 2, 3)) / total_voxels
        y_centroid = torch.sum(yy * volume, dim=(1, 2, 3)) / total_voxels
        x_centroid = torch.sum(xx * volume, dim=(1, 2, 3)) / total_voxels

        centroids = torch.stack([z_centroid, y_centroid, x_centroid], dim=1).squeeze()
        return centroids

    pred_centroids = compute_centroids(pred)
    gt_centroids = compute_centroids(gt)
    loss = F.mse_loss(pred_centroids, gt_centroids)
    return loss


def mse_patch_loss(pred, gt, k):
    # Define a patch extraction function using 3D convolution
    def extract_patches(tensor, k):
        # Create a uniform kernel of ones
        kernel = torch.ones(1, 1, k, k, k).to(tensor.device)
        kernel = kernel / kernel.numel()
        # Extract patches using 3D convolution
        patches = F.conv3d(
            tensor.unsqueeze(1), kernel, stride=k // 2, padding=0
        ).squeeze()
        return patches

    # Extract patches
    pred_patches = extract_patches(pred, k)
    gt_patches = extract_patches(gt, k)

    # Ensure the size matches
    assert (
        pred_patches.shape == gt_patches.shape
    ), "Mismatch in patch sizes. Make sure input tensors have the same shape."

    # Compute MSE for the maximum values of the overlapping patches
    loss = F.mse_loss(pred_patches, gt_patches)
    return loss


def shape_loss(pred, gt):
    def otsu_threshold(gray_img, nbins=0.1):
        all_pixels = gray_img.flatten()
        p_all = len(all_pixels)
        least_variance = -1
        least_variance_threshold = -1

        # create an array of all possible threshold values which we want to loop through
        color_thresholds = torch.arange(
            gray_img.min().item() + nbins, gray_img.max().item() - nbins, nbins
        )

        # loop through the thresholds to find the one with the least class variance
        for color_threshold in color_thresholds:
            # background
            bg_pixels = all_pixels[all_pixels < color_threshold]
            p_bg = len(bg_pixels)
            w_bg = p_bg / p_all
            variance_bg = torch.var(bg_pixels)

            # foreground
            fg_pixels = all_pixels[all_pixels >= color_threshold]
            p_fg = len(fg_pixels)
            w_fg = p_fg / p_all
            variance_fg = torch.var(fg_pixels)

            variance = w_bg * variance_bg + w_fg * variance_fg

            if least_variance == -1 or variance < least_variance:
                least_variance = variance
                least_variance_threshold = color_threshold
        return least_variance_threshold

    # Apply Otsu's thresholding
    pred_threshold = torch.quantile(pred.flatten(start_dim=1), 0.5, dim=1).view(
        -1, 1, 1, 1
    )
    gt_threshold = torch.quantile(gt.flatten(start_dim=1), 0.5, dim=1).view(-1, 1, 1, 1)
    # pred_threshold = torch.stack([otsu_threshold(s.amax(0)) for s in pred]).view(-1, 1, 1, 1).to(gt.device)
    # gt_threshold = torch.stack([otsu_threshold(s.amax(0)) for s in gt]).view(-1, 1, 1, 1).to(gt.device)

    pred_bin = pred > pred_threshold
    gt_bin = gt > gt_threshold

    # Compute the bounding box dimensions
    def compute_dimensions(tensor):
        non_zeros = tensor.nonzero(as_tuple=True)
        min_coords = [torch.min(coord) for coord in non_zeros]
        max_coords = [torch.max(coord) for coord in non_zeros]

        width = (max_coords[3] - min_coords[3]) / tensor.shape[3]
        height = (max_coords[2] - min_coords[2]) / tensor.shape[2]
        depth = (max_coords[1] - min_coords[1]) / tensor.shape[1]

        return torch.stack([depth, height, width], dim=-1).float()

    pred_dims = compute_dimensions(pred_bin)
    gt_dims = compute_dimensions(gt_bin)

    # Compute the MSE loss between bounding box dimensions
    loss = F.mse_loss(pred_dims, gt_dims)
    return loss


def energy_loss(pred, gt):
    def compute_energy(t):
        return (t**2).sum(dim=(2, 3)).mean(1)

    pred_energy = compute_energy(pred)
    gt_energy = compute_energy(gt)

    loss = F.mse_loss(pred_energy, gt_energy)
    return loss


def ssim_loss(pred, gt):
    ssim_score = torchmetrics.functional.image.structural_similarity_index_measure(
        preds=pred,
        target=gt,
    )
    return 1 - ssim_score


def plot_images_with_centroids(
    image1,
    image2,
    adjusted_image2,
):
    """Plot the images and their centroids."""
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    axes[0, 0].imshow(image1, cmap="coolwarm")
    axes[0, 0].set_title("Original image 1")

    axes[0, 1].imshow(image2, cmap="coolwarm")
    axes[0, 1].set_title("Original image 2")

    axes[1, 0].imshow(adjusted_image2, cmap="coolwarm")
    axes[1, 0].set_title("Interpolated image 2")

    axes[1, 1].imshow(image1 - image2, cmap="coolwarm")
    axes[1, 1].set_title("Differences without interpolation")

    plt.savefig("centroids.png")


if __name__ == "__main__":
    # Example usage
    # Assuming pred and gt are your fMRI images, each of shape [Height, Width]
    import random
    from datasets.eeg_fmri_preprocessed import NoddiPreprocessedDataset

    dataset_path = f"../../datasets/eeg_fmri_prep_512hz"
    dataset = NoddiPreprocessedDataset(
        path=dataset_path,
        normalize_eegs=True,
        normalize_fmris=True,
    )
    gt = torch.stack(
        [
            torch.from_numpy(dataset[random.randint(0, len(dataset))]["fmris"])
            for _ in range(8)
        ]
    )
    pred = gt.clone()

    class ImageAdjustment(torch.nn.Module):
        def __init__(self, image_size):
            super(ImageAdjustment, self).__init__()
            batch_size = image_size[0]
            self.offset = torch.nn.Parameter(
                torch.zeros(batch_size, 3), requires_grad=True
            )  # [batch_size, 3]
            self.image_size = image_size[1:]

        def forward(self, image):
            # Assuming image is of shape [batch_size, depth, height, width]
            batch_size = image.size(0)

            # Create a grid to apply the offset
            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, self.image_size[0]),
                torch.linspace(-1, 1, self.image_size[1]),
                torch.linspace(-1, 1, self.image_size[2]),
                indexing="ij",
            )
            grid = torch.stack([grid_x, grid_y, grid_z], dim=-1).to(
                image.device
            )  # [depth, height, width, 3]

            # Clamp the offset values for each dimension
            clamped_offset_x = torch.clamp(
                self.offset[:, 0],
                min=-self.image_size[2] / 4,
                max=self.image_size[2] / 4,
            )
            clamped_offset_y = torch.clamp(
                self.offset[:, 1],
                min=-self.image_size[1] / 4,
                max=self.image_size[1] / 4,
            )
            clamped_offset_z = torch.clamp(
                self.offset[:, 2],
                min=-self.image_size[0] / 4,
                max=self.image_size[0] / 4,
            )
            clamped_offset = torch.stack(
                [clamped_offset_x, clamped_offset_y, clamped_offset_z], dim=-1
            )  # [batch_size, 3]

            # Adjust grid for each image in the batch
            grid = grid.unsqueeze(0) + clamped_offset.view(
                batch_size, 1, 1, 1, 3
            )  # Broadcasting

            # Use grid_sample to move the image by the offset
            return F.grid_sample(
                image.unsqueeze(
                    1
                ),  # Add channel dim: [batch_size, 1, depth, height, width]
                grid,
                align_corners=True,
                mode="bilinear",
            ).squeeze(
                1
            )  # Remove channel dim

    adjuster_pred = ImageAdjustment(image_size=pred.shape).to(pred.device)
    adjuster_pred.offset = torch.nn.Parameter(torch.rand(pred.shape[0], 3) * 0.2)
    pred = adjuster_pred(pred).detach()

    loss_normal = F.mse_loss(pred, gt)
    print(f"Normal MSE Loss: {loss_normal.item()}")

    loss_adj, pred_adj = custom_loss(pred, gt, return_adjusted=True)
    # loss_adj = custom_loss(pred, gt, return_adjusted=False)
    loss_adj.backward()
    print(f"Adjusted MSE Loss: {loss_adj.item()}")

    print(f"centroid loss", centroid_loss(pred, gt))
    print(f"centroid loss between self", centroid_loss(pred, pred))

    loss_shape = shape_loss(pred, gt)
    print(f"shape loss", loss_shape)

    loss_energy = energy_loss(pred, gt)
    print(f"shape energy", loss_energy)

    loss_energy = mse_patch_loss(pred, gt, k=4)
    print(f"shape mse_patch_loss", loss_energy)

    loss_ssim = ssim_loss(pred, pred_adj)
    print(f"shape ssim loss", loss_ssim)

    pred_mip = pred.amax(dim=1)[0]
    gt_mip = gt.amax(dim=1)[0]
    pred_adj_mip = pred_adj.amax(dim=1)[0]
    plot_images_with_centroids(
        image1=gt_mip.numpy(),
        image2=pred_mip.numpy(),
        adjusted_image2=pred_adj_mip.detach().numpy(),
    )
