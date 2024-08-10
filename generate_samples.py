try:
    import torch
    import torchvision.utils as vutils
    from tqdm import tqdm
    from lib.networks import VAE
    from pathlib import Path
    from lib.utils import Utils
    import hydra
    import cv2
    from lib.networks import padding_image
    import numpy as np

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def get_checkpoint(models_dir: Path) -> str:
    if not models_dir.is_dir():
        raise Exception(f"{models_dir} is not a directory!")

    ckpts = [
        f
        for f in models_dir.iterdir()
        if (f.suffix.lower() == ".pt" and f.stem.lower().startswith("val_"))
    ]

    return str(ckpts[0])


def load_image(image_path: Path, resolution: int):
    img = cv2.imread(str(image_path), 0)
    img = padding_image(img, resolution)
    img = cv2.bitwise_not(img)
    img = np.asarray(img, float) / 255.0

    return torch.from_numpy(np.expand_dims(np.expand_dims(img.copy(), 0), 0)).float()


@hydra.main(version_base="1.2", config_path="config", config_name="cfg")
def generate_test_set(args):
    batch_size = args.train.batch_size
    image_size = args.train.image_size

    Path(args.model_out_path).mkdir(exist_ok=True, parents=True)
    Path(args.results_out_path).mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader, valloader = Utils.get_dataloaders(
        dataset_path=Path(args.dataset_path),
        batch_size=batch_size,
        resolution=image_size,
    )

    # Create VAE network
    vae_net = VAE(
        channel_in=1,
        ch=args.vae.ch,
        blocks=tuple(args.vae.blocks),
        latent_channels=args.vae.latent_channels,
    ).to(device)

    # Carica il file salvato
    checkpoint = torch.load(
        get_checkpoint(Path(args.model_out_path)), weights_only=True
    )
    # Ripristina lo stato del modello e dell'ottimizzatore
    vae_net.load_state_dict(checkpoint["model_state_dict"])

    vae_net.eval()
    pbar_batches = tqdm(total=len(testloader), desc="Batch", leave=False)
    for i, images in enumerate(testloader):
        images = images.to(device)

        _, mu, logvar = vae_net(images)

        # Reparametrization trick per campionare dallo spazio latente
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        z = mu + std * epsilon

        print(z.shape)
        print(torch.max(z))
        print(torch.min(z))

        recon_img = vae_net.decoder(z)
        img_cat = torch.cat((recon_img.cpu(), images.cpu()), 2)
        vutils.save_image(
            img_cat,
            f"{args.results_out_path}/generated_{i}.png",
            normalize=True,
        )
        break

        pbar_batches.update(1)
    pbar_batches.close()


@hydra.main(version_base="1.2", config_path="config", config_name="cfg")
def generate_single_image(args):
    Path(args.model_out_path).mkdir(exist_ok=True, parents=True)
    Path(args.results_out_path).mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img = load_image(
        "/home/tom/git_workspace/CVAE_Graphene/DefectedGraphene/data/training_dataset2/test/graphene_152268_opt.png",
        resolution=args.train.image_size,
    )
    img = img.to(device)
    print(img.shape)

    # Create VAE network
    vae_net = VAE(
        channel_in=1,
        ch=args.vae.ch,
        blocks=tuple(args.vae.blocks),
        latent_channels=args.vae.latent_channels,
    ).to(device)

    # Carica il file salvato
    checkpoint = torch.load(
        get_checkpoint(Path(args.model_out_path)), weights_only=True
    )
    # Ripristina lo stato del modello e dell'ottimizzatore
    vae_net.load_state_dict(checkpoint["model_state_dict"])

    vae_net.eval()

    _, mu, logvar = vae_net(img)

    # Reparametrization trick per campionare dallo spazio latente
    std = torch.exp(0.5 * logvar)

    scale_factor = 1.5
    epsilon = torch.randn_like(std) * scale_factor

    noise_factor = 0.1
    mu_noisy = mu + noise_factor * torch.randn_like(mu)

    z = mu_noisy + std * epsilon

    print(z.shape)
    print(torch.max(z))
    print(torch.min(z))

    recon_img = vae_net.decoder(z)

    vutils.save_image(
        recon_img.cpu(),
        f"{args.results_out_path}/single_generated.png",
        normalize=True,
    )
    vutils.save_image(
        img.cpu(),
        f"{args.results_out_path}/single_original.png",
        normalize=True,
    )

    # Utils.sharpened_image(Path(f"{args.results_out_path}/single_generated.png"))


if __name__ == "__main__":
    generate_single_image()
