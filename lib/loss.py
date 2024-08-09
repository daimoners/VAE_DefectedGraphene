try:
    import torch
    import torch.nn as nn

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def kl_loss(mu, logvar):
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()


def image_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images"""
    ssim = SSIM()

    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    ssim_loss = ssim.forward(pred, target).mean(1, True)

    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

    return reprojection_loss.mean()


def fourier_loss(recon_img, original_img):
    """
    Calcola la perdita basata sulla differenza tra le trasformate di Fourier
    delle immagini ricostruite e originali.

    Args:
        recon_img (Tensor): Immagini ricostruite dal modello.
        original_img (Tensor): Immagini originali di input.

    Returns:
        Tensor: La perdita calcolata nel dominio delle frequenze.
    """
    # Trasformata di Fourier 2D per le immagini ricostruite e originali
    recon_fft = torch.fft.fftn(recon_img, dim=(-2, -1))
    original_fft = torch.fft.fftn(original_img, dim=(-2, -1))

    # Calcolo della differenza nel dominio delle frequenze
    # Utilizzando L1 norm
    fourier_loss_value = torch.mean(torch.abs(recon_fft - original_fft))

    return fourier_loss_value


if __name__ == "__main__":
    pass
