import torch


def _compute_kernel(x, y):
    x_size = x.shape[0]
    y_size = y.shape[0]
    dim = x.shape[1]

    tiled_x = x.view(x_size, 1, dim).repeat(1, y_size, 1)
    tiled_y = y.view(1, y_size, dim).repeat(x_size, 1, 1)

    return torch.exp(-torch.mean((tiled_x - tiled_y) ** 2, dim=2) / dim * 1.0)


def _compute_mmd(x, y):
    x_kernel = _compute_kernel(x, x)
    y_kernel = _compute_kernel(y, y)
    xy_kernel = _compute_kernel(x, y)
    return torch.mean(x_kernel) + torch.mean(y_kernel) - 2 * torch.mean(xy_kernel)


def mmd_loss(latent):
    true_samples = torch.randn(latent.size(), device=latent.device)
    return _compute_mmd(true_samples, latent)


REG_LOSSES = {
    'MMD': mmd_loss,
}
