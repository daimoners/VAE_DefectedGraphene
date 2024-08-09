try:
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    from tqdm import tqdm
    from lib.networks import VGG19, MyDatasetPng, VAE
    from pathlib import Path
    import numpy as np
    import torch.optim.lr_scheduler as Sched
    from lib.loss import image_loss, fourier_loss, kl_loss
    from lib.utils import Utils
    import hydra
    from telegram_bot import send_images, send_message

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def get_dataloaders(dataset_path: Path, batch_size: int, resolution: int):
    if not dataset_path.is_dir():
        raise Exception(f"{dataset_path} is not a directory!")

    train_path = dataset_path.joinpath("train")
    val_path = dataset_path.joinpath("val")
    test_path = dataset_path.joinpath("test")

    data_transform = transforms.Compose(
        [
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ]
    )

    train_paths = [
        f for f in train_path.iterdir() if f.suffix.lower() in Utils.IMAGE_EXTENSIONS
    ]
    test_paths = [
        f for f in test_path.iterdir() if f.suffix.lower() in Utils.IMAGE_EXTENSIONS
    ]
    val_paths = [
        f for f in val_path.iterdir() if f.suffix.lower() in Utils.IMAGE_EXTENSIONS
    ]

    train_data = MyDatasetPng(
        train_paths, resolution=resolution, transforms=data_transform
    )
    test_data = MyDatasetPng(
        test_paths, resolution=resolution, transforms=data_transform
    )
    val_data = MyDatasetPng(val_paths, resolution=resolution, transforms=data_transform)

    trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    return trainloader, testloader, valloader


@hydra.main(version_base="1.2", config_path="config", config_name="cfg")
def main(args):
    batch_size = args.train.batch_size
    image_size = args.train.image_size
    lr = args.train.lr
    nepoch = args.train.nepoch
    early_stop_patience = args.train.early_stop_patience
    scheduler_patience = args.train.scheduler_patience

    Path(args.model_out_path).mkdir(exist_ok=True, parents=True)
    Path(args.results_out_path).mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, testloader, valloader = get_dataloaders(
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

    # Feature extractor
    feature_extractor = VGG19().to(device)

    # setup optimizer
    optimizer = optim.Adam(vae_net.parameters(), lr=lr, betas=(0.5, 0.999))
    # setup scheduler
    scheduler = Sched.ReduceLROnPlateau(
        optimizer, mode="min", patience=scheduler_patience
    )
    # Loss function
    loss_log = []
    val_loss_log = []

    lowest_val_loss = np.inf

    dataiter = iter(testloader)
    test_images = next(dataiter)
    test_images.shape

    # TRAIN
    early_stop_count = 0
    pbar_epochs = tqdm(total=nepoch, desc="Epochs", leave=False)
    for epoch in range(0, nepoch):
        vae_net.train()
        pbar_batches = tqdm(total=len(trainloader), desc="Batch", leave=False)
        for images in trainloader:
            images = images.to(device)

            recon_img, mu, logvar = vae_net(images)
            # VAE loss
            kl_loss_value = kl_loss(mu, logvar)

            image_loss_value = image_loss(recon_img, images)
            # Perception loss
            feat_in = torch.cat(
                (Utils.grayscale_to_rgb(recon_img), Utils.grayscale_to_rgb(images)), 0
            )
            feature_loss_value = feature_extractor(feat_in)

            loss = (
                kl_loss_value
                + image_loss_value
                + feature_loss_value
                + fourier_loss(recon_img, images)
            )

            # print(loss)

            loss_log.append(loss.item())
            vae_net.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_batches.update(1)
            pbar_batches.set_postfix(loss=loss.item())
        pbar_batches.close()

        # In eval mode the model will use mu as the encoding instead of sampling from the distribution
        vae_net.eval()
        pbar_batches = tqdm(total=len(valloader), desc="Batch", leave=False)
        with torch.no_grad():
            for images in valloader:
                images = images.to(device)

                recon_img, mu, logvar = vae_net(images)
                # VAE loss
                kl_loss_value = kl_loss(mu, logvar)
                # mse_loss = F.mse_loss(recon_img, images)
                image_loss_value = image_loss(recon_img, images)
                # Perception loss
                feat_in = torch.cat(
                    (Utils.grayscale_to_rgb(recon_img), Utils.grayscale_to_rgb(images)),
                    0,
                )
                feature_loss_value = feature_extractor(feat_in)

                loss = (
                    kl_loss_value
                    + image_loss_value
                    + feature_loss_value
                    + fourier_loss(recon_img, images)
                )

                val_loss_log.append(loss.item())

                pbar_batches.update(1)
                pbar_batches.set_postfix(loss=loss.item())
        pbar_batches.close()

        scheduler.step(np.mean(val_loss_log))

        if np.mean(val_loss_log) < lowest_val_loss:
            Utils.remove_old_models(Path("./Models/"))
            torch.save(
                {
                    "epoch": epoch,
                    "loss_log": loss_log,
                    "model_state_dict": vae_net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"{args.model_out_path}/val_loss_{np.mean(val_loss_log):.4f}.pt",
            )
            lowest_val_loss = np.mean(val_loss_log)
            early_stop_count = 0

            recon_img, _, _ = vae_net(test_images.to(device))
            img_cat = torch.cat((recon_img.cpu(), test_images), 2)

            vutils.save_image(
                img_cat,
                f"{args.results_out_path}/img_temp.png",
                normalize=True,
            )
        else:
            early_stop_count += 1
            print(f"\nEarly stop count: {early_stop_count}")

        if early_stop_count >= early_stop_patience:
            print(f"\nEarly stop patience reached, stop the training")
            break

        pbar_epochs.update(1)
        pbar_epochs.set_postfix(mean_loss=np.mean(val_loss_log))
    pbar_epochs.close()

    # TEST
    # Carica il file salvato
    checkpoint = torch.load(
        f"{args.model_out_path}/val_loss_{lowest_val_loss:.4f}.pt", weights_only=True
    )
    # Ripristina lo stato del modello e dell'ottimizzatore
    vae_net.load_state_dict(checkpoint["model_state_dict"])
    vae_net.eval()
    with torch.no_grad():
        recon_img, _, _ = vae_net(test_images.to(device))
        img_cat = torch.cat((recon_img.cpu(), test_images), 2)

        vutils.save_image(
            img_cat,
            f"{args.results_out_path}/img_{lowest_val_loss:.4f}.png",
            normalize=True,
        )

    if args.telegram_bot:
        message = f"Terminated VAE training with lowest val loss: {lowest_val_loss}"
        send_message(message, parse_mode="MarkdownV2", disable_notification=True)
        send_images(
            {
                "VAE Results": Path(
                    f"{args.results_out_path}/img_{lowest_val_loss:.4f}.png"
                )
            },
        )


if __name__ == "__main__":
    main()
