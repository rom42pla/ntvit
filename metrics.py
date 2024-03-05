import torch
from torch.nn import functional as F
import torchmetrics

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
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        ).squeeze(
            1
        )  # Remove channel dim
        
def warped_metric(pred, gt, metric_fn, mode="max", max_iterations=100, lr=1e-2):
    assert mode in {"min", "max"}, f"got {mode}"
    adjuster = ImageAdjustment(image_size=pred.shape).to(pred.device)
    optimizer = torch.optim.AdamW(adjuster.parameters(), lr=lr, maximize=True if mode == "max" else False)
    best_score = torch.inf if mode == "min" else -torch.inf

    for _ in range(max_iterations):
        with torch.enable_grad():
            optimizer.zero_grad()
            score = metric_fn(adjuster(pred), gt.detach())
            if (mode == "max" and score > best_score) or (mode == "min" and score < best_score):
                best_score = score.clone().detach()
            score.backward()
            optimizer.step()
    return best_score

def psnr(pred, gt):
    return torchmetrics.functional.image.peak_signal_noise_ratio(
            preds=pred,
            target=gt,
        )

def ssim(pred, gt):
    return torchmetrics.functional.image.structural_similarity_index_measure(
            preds=pred,
            target=gt,
        )
    
def rmse(pred, gt):
    return torchmetrics.functional.regression.mean_squared_error(
            preds=pred,
            target=gt,
            squared=False
        )
    
def mae(pred, gt):
    return torchmetrics.functional.regression.mean_absolute_error(
            preds=pred,
            target=gt,
        )
    
def cfv(pred, gt):
    return F.cosine_similarity(pred.flatten(start_dim=1), gt.flatten(start_dim=1)).mean()

def lcfv(pred, gt):
    return torch.log(1 - F.cosine_similarity(pred.flatten(start_dim=1), gt.flatten(start_dim=1))).mean()
    
#######################
# TESTS
#######################
def test_psnr():
    pred, gt = torch.randn([32, 32, 64, 64]).chunk(2)
    assert psnr(pred, gt) >= 0
    

def test_ssim():
    pred, gt = torch.randn([32, 32, 64, 64]).chunk(2)
    assert 0 <= ssim(pred, gt) <= 1
    
def test_rmse():
    pred, gt = torch.randn([32, 32, 64, 64]).chunk(2)
    assert rmse(pred, gt) >= 0

def test_mae():
    pred, gt = torch.randn([32, 32, 64, 64]).chunk(2)
    assert mae(pred, gt) >= 0

def test_cfv():
    pred, gt = torch.randn([32, 32, 64, 64]).chunk(2)
    assert cfv(pred, gt)
    
def test_lcfv():
    pred, gt = torch.randn([32, 32, 64, 64]).chunk(2)
    assert lcfv(pred, gt)
    
if __name__ == "__main__":
    import random
    from datasets.eeg_fmri_preprocessed import NoddiPreprocessedDataset
    dataset_path = f"../../datasets/eeg_fmri_prep_512hz"
    dataset = NoddiPreprocessedDataset(
        path=dataset_path,
        normalize_eegs=True,
        normalize_fmris=True,
    )
    pred, gt = torch.stack(
        [
            torch.from_numpy(dataset[random.randint(0, len(dataset))]["fmris"])
            for _ in range(16*2)
        ]
    ).cuda().chunk(2)
    assert pred.shape == gt.shape
    
    print("psnr", psnr(pred, gt))
    print("warped psnr", warped_metric(pred, gt, metric_fn=psnr, mode="max"))
    print('mse', F.mse_loss(pred, gt))
    print("warped mse", warped_metric(pred, gt, metric_fn=F.mse_loss, mode="min"))
    # print(cfv(pred, gt))
    # print(lcfv(pred, gt))