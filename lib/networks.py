try:
    import torch.nn as nn
    import torch
    import cv2
    import numpy as np
    from PIL import Image

except Exception as e:
    print(f"Some module are missing: {e}")


def padding_image(image):
    h = image.shape[0]
    w = image.shape[1]

    if h >= w:
        size = h + 1
    else:
        size = w + 1

    top = round((size - h) / 2)
    bottom = size - (h + top)

    left = round((size - w) / 2)
    right = size - (w + left)

    padded_img = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0
    )

    return padded_img


class MyDatasetPng:
    """Class that generate a dataset for DataLoader module, given as input the paths of the .png files and the respective labels"""

    def __init__(self, paths, resolution, transforms):
        self.paths = paths
        self.resolution = resolution
        self.transform = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]), 0)
        img = padding_image(img)
        img = cv2.bitwise_not(img)
        img = np.asarray(img, float) / 255.0

        # Converti l'immagine da numpy array a PIL Image
        if len(img.shape) == 2:
            img = Image.fromarray(
                (img * 255).astype(np.uint8), mode="L"
            )  # Immagine in scala di grigi
        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = Image.fromarray(
                (img * 255).astype(np.uint8), mode="RGB"
            )  # Immagine RGB
        else:
            raise Exception("Wrong dimensions for the input images\n")

        # Applica le trasformazioni se definite
        if self.transform:
            img = self.transform(img)

        return img


## Feature extractor
class VGG19(nn.Module):
    """
    Simplified version of the VGG19 "feature" block
    This module's only job is to return the "feature loss" for the inputs
    """

    def __init__(self, channel_in=3, width=64):
        super(VGG19, self).__init__()

        self.conv1 = nn.Conv2d(channel_in, width, 3, 1, 1)
        self.conv2 = nn.Conv2d(width, width, 3, 1, 1)

        self.conv3 = nn.Conv2d(width, 2 * width, 3, 1, 1)
        self.conv4 = nn.Conv2d(2 * width, 2 * width, 3, 1, 1)

        self.conv5 = nn.Conv2d(2 * width, 4 * width, 3, 1, 1)
        self.conv6 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv7 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)
        self.conv8 = nn.Conv2d(4 * width, 4 * width, 3, 1, 1)

        self.conv9 = nn.Conv2d(4 * width, 8 * width, 3, 1, 1)
        self.conv10 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv11 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv12 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.conv13 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv14 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv15 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)
        self.conv16 = nn.Conv2d(8 * width, 8 * width, 3, 1, 1)

        self.mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        self.load_params_()

    def load_params_(self):
        # Download and load Pytorch's pre-trained weights
        state_dict = torch.hub.load_state_dict_from_url(
            "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth"
        )
        # # Remove the first layer's weights
        # state_dict.pop("features.0.weight")
        # state_dict.pop("features.0.bias")

        # Load the rest of the state_dict
        own_state_dict = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)
                own_state_dict[name].requires_grad = False

    def feature_loss(self, x):
        return (x[: x.shape[0] // 2] - x[x.shape[0] // 2 :]).pow(2).mean()

    def forward(self, x):
        """
        :param x: Expects x to be the target and source to concatenated on dimension 0
        :return: Feature loss
        """
        x = self.conv1(x)
        loss = self.feature_loss(x)
        x = self.conv2(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 64x64

        x = self.conv3(x)
        loss += self.feature_loss(x)
        x = self.conv4(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 32x32

        x = self.conv5(x)
        loss += self.feature_loss(x)
        x = self.conv6(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv7(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv8(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 16x16

        x = self.conv9(x)
        loss += self.feature_loss(x)
        x = self.conv10(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv11(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv12(self.relu(x))
        loss += self.feature_loss(x)
        x = self.mp(self.relu(x))  # 8x8

        x = self.conv13(x)
        loss += self.feature_loss(x)
        x = self.conv14(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv15(self.relu(x))
        loss += self.feature_loss(x)
        x = self.conv16(self.relu(x))
        loss += self.feature_loss(x)

        return loss / 16


# old VAE
# class VAE(nn.Module):
#     def __init__(self, latent_dim=20):
#         super(VAE, self).__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Conv2d(
#                 1, 32, kernel_size=4, stride=2, padding=1
#             ),  # Riduce la dimensione a metà
#             nn.ReLU(),
#             nn.Conv2d(
#                 32, 64, kernel_size=4, stride=2, padding=1
#             ),  # Riduce la dimensione a metà
#             nn.ReLU(),
#             nn.Conv2d(
#                 64, 128, kernel_size=4, stride=2, padding=1
#             ),  # Riduce la dimensione a metà
#             nn.ReLU(),
#             nn.Conv2d(
#                 128, 256, kernel_size=4, stride=2, padding=1
#             ),  # Riduce la dimensione a metà
#             nn.ReLU(),
#         )

#         # Adaptive pooling per garantire un output di dimensione fissa 4x4
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

#         # Definizione dei fully connected layers
#         self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
#         self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

#         # Decoder
#         self.fc_decode = nn.Linear(latent_dim, 256 * 4 * 4)

#         # Convoluzioni trasposte nel decoder
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(
#                 256, 128, kernel_size=4, stride=2, padding=1
#             ),  # Scala la dimensione a 8x8
#             nn.ReLU(),
#             nn.ConvTranspose2d(
#                 128, 64, kernel_size=4, stride=2, padding=1
#             ),  # Scala la dimensione a 16x16
#             nn.ReLU(),
#             nn.ConvTranspose2d(
#                 64, 32, kernel_size=4, stride=2, padding=1
#             ),  # Scala la dimensione a 32x32
#             nn.ReLU(),
#             nn.ConvTranspose2d(
#                 32, 16, kernel_size=4, stride=2, padding=1
#             ),  # Scala la dimensione a 64x64
#             nn.ReLU(),
#             nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),  # Riduce i canali a 1
#             nn.Sigmoid(),
#         )

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         # Encoder
#         batch_size, _, height, width = x.size()  # Ottieni la dimensione dell'input
#         x = self.encoder(x)
#         x = self.adaptive_pool(x)  # Ridimensiona le feature map a 4x4
#         x = x.view(x.size(0), -1)
#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)
#         z = self.reparameterize(mu, logvar)

#         # Decoder
#         x = self.fc_decode(z)
#         x = x.view(x.size(0), 256, 4, 4)  # Riforma il tensor per il decoder
#         x = self.decoder(x)

#         # Adatta l'output alla dimensione dell'input usando Upsample
#         x = F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)

#         return x, mu, logvar


## VAE
def get_norm_layer(channels, norm_type="bn"):
    if norm_type == "bn":
        return nn.BatchNorm2d(channels, eps=1e-4)
    elif norm_type == "gn":
        return nn.GroupNorm(8, channels, eps=1e-4)
    else:
        ValueError("norm_type must be bn or gn")


class ResDown(nn.Module):
    """
    Residual down sampling block for the encoder
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResDown, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(
            channel_in,
            (channel_out // 2) + channel_out,
            kernel_size,
            2,
            kernel_size // 2,
        )
        self.norm2 = get_norm_layer(channel_out // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(
            channel_out // 2, channel_out, kernel_size, 1, kernel_size // 2
        )

        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x):
        x = self.act_fnc(self.norm1(x))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResUp(nn.Module):
    """
    Residual up sampling block for the decoder
    """

    def __init__(
        self, channel_in, channel_out, kernel_size=3, scale_factor=2, norm_type="bn"
    ):
        super(ResUp, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        self.conv1 = nn.Conv2d(
            channel_in,
            (channel_in // 2) + channel_out,
            kernel_size,
            1,
            kernel_size // 2,
        )
        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(
            channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2
        )

        self.up_nn = nn.Upsample(scale_factor=scale_factor, mode="nearest")
        self.act_fnc = nn.ELU()
        self.channel_out = channel_out

    def forward(self, x_in):
        x = self.up_nn(self.act_fnc(self.norm1(x_in)))

        # Combine skip and first conv into one layer for speed
        x_cat = self.conv1(x)
        skip = x_cat[:, : self.channel_out]
        x = x_cat[:, self.channel_out :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class ResBlock(nn.Module):
    """
    Residual block
    """

    def __init__(self, channel_in, channel_out, kernel_size=3, norm_type="bn"):
        super(ResBlock, self).__init__()
        self.norm1 = get_norm_layer(channel_in, norm_type=norm_type)

        first_out = (
            channel_in // 2
            if channel_in == channel_out
            else (channel_in // 2) + channel_out
        )
        self.conv1 = nn.Conv2d(channel_in, first_out, kernel_size, 1, kernel_size // 2)

        self.norm2 = get_norm_layer(channel_in // 2, norm_type=norm_type)

        self.conv2 = nn.Conv2d(
            channel_in // 2, channel_out, kernel_size, 1, kernel_size // 2
        )
        self.act_fnc = nn.ELU()
        self.skip = channel_in == channel_out
        self.bttl_nk = channel_in // 2

    def forward(self, x_in):
        x = self.act_fnc(self.norm1(x_in))

        x_cat = self.conv1(x)
        x = x_cat[:, : self.bttl_nk]

        # If channel_in == channel_out we do a simple identity skip
        if self.skip:
            skip = x_in
        else:
            skip = x_cat[:, self.bttl_nk :]

        x = self.act_fnc(self.norm2(x))
        x = self.conv2(x)

        return x + skip


class Encoder(nn.Module):
    """
    Encoder block
    """

    def __init__(
        self,
        channels,
        ch=64,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(channels, blocks[0] * ch, 3, 1, 1)

        widths_in = list(blocks)
        widths_out = list(blocks[1:]) + [2 * blocks[-1]]

        self.layer_blocks = nn.ModuleList([])
        for w_in, w_out in zip(widths_in, widths_out):

            if deep_model:
                # Add an additional non down-sampling block before down-sampling
                self.layer_blocks.append(
                    ResBlock(w_in * ch, w_in * ch, norm_type=norm_type)
                )

            self.layer_blocks.append(
                ResDown(w_in * ch, w_out * ch, norm_type=norm_type)
            )

        for _ in range(num_res_blocks):
            self.layer_blocks.append(
                ResBlock(widths_out[-1] * ch, widths_out[-1] * ch, norm_type=norm_type)
            )

        self.conv_mu = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.conv_log_var = nn.Conv2d(widths_out[-1] * ch, latent_channels, 1, 1)
        self.act_fnc = nn.ELU()

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, sample=False):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        mu = self.conv_mu(x)
        log_var = self.conv_log_var(x)

        if self.training or sample:
            x = self.sample(mu, log_var)
        else:
            x = mu

        return x, mu, log_var


class Decoder(nn.Module):
    """
    Decoder block
    Built to be a mirror of the encoder block
    """

    def __init__(
        self,
        channels,
        ch=64,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(Decoder, self).__init__()
        widths_out = list(blocks)[::-1]
        widths_in = (list(blocks[1:]) + [2 * blocks[-1]])[::-1]

        self.conv_in = nn.Conv2d(latent_channels, widths_in[0] * ch, 1, 1)

        self.layer_blocks = nn.ModuleList([])
        for _ in range(num_res_blocks):
            self.layer_blocks.append(
                ResBlock(widths_in[0] * ch, widths_in[0] * ch, norm_type=norm_type)
            )

        for w_in, w_out in zip(widths_in, widths_out):
            self.layer_blocks.append(ResUp(w_in * ch, w_out * ch, norm_type=norm_type))
            if deep_model:
                # Add an additional non up-sampling block after up-sampling
                self.layer_blocks.append(
                    ResBlock(w_out * ch, w_out * ch, norm_type=norm_type)
                )

        self.conv_out = nn.Conv2d(blocks[0] * ch, channels, 5, 1, 2)
        self.act_fnc = nn.ELU()

    def forward(self, x):
        x = self.conv_in(x)

        for block in self.layer_blocks:
            x = block(x)
        x = self.act_fnc(x)

        return torch.tanh(self.conv_out(x))


class VAE(nn.Module):
    """
    VAE network, uses the above encoder and decoder blocks
    """

    def __init__(
        self,
        channel_in=1,
        ch=64,
        blocks=(1, 2, 4, 8),
        latent_channels=256,
        num_res_blocks=1,
        norm_type="bn",
        deep_model=False,
    ):
        super(VAE, self).__init__()
        """Res VAE Network
        channel_in  = number of channels of the image 
        z = the number of channels of the latent representation
        (for a 64x64 image this is the size of the latent vector)
        """

        self.encoder = Encoder(
            channel_in,
            ch=ch,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )
        self.decoder = Decoder(
            channel_in,
            ch=ch,
            blocks=blocks,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            norm_type=norm_type,
            deep_model=deep_model,
        )

    def forward(self, x):
        encoding, mu, log_var = self.encoder(x)
        recon_img = self.decoder(encoding)
        return recon_img, mu, log_var


if __name__ == "__main__":
    pass
