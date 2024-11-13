try:
    import torch
    import torch.optim as optim
    import torchvision.utils as vutils
    from tqdm import tqdm
    from lib.networks import VGG19, VAE
    from pathlib import Path
    import numpy as np
    import torch.optim.lr_scheduler as Sched
    from lib.loss import image_loss, fourier_loss, kl_loss
    from lib.utils import Utils
    import hydra
    from telegram_bot import send_images, send_message
    from icecream import ic
    import submitit
    from omegaconf import open_dict, OmegaConf
    import pandas as pd

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        train(self.args)


@hydra.main(version_base="1.2", config_path="config", config_name="cfg")
def main(args):
    if args.verbose:
        ic.enable()
    else:
        ic.disable()

    if args.deterministic:
        torch.manual_seed(42 if not args.random_seed else args.random_seed)
        torch.cuda.manual_seed(42 if not args.random_seed else args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(42 if not args.random_seed else args.random_seed)

    if args.matmul_precision == "highest":
        torch.set_float32_matmul_precision("highest")
    elif args.matmul_precision == "high":
        torch.set_float32_matmul_precision("high")
    elif args.matmul_precision == "medium":
        torch.set_float32_matmul_precision("medium")

    if args.slurm:
        Path(args.slurm_output).mkdir(parents=True, exist_ok=True)
        executor = submitit.AutoExecutor(
            folder=args.slurm_output,
            slurm_max_num_timeout=30,
        )

        executor.update_parameters(
            mem_gb=12 * args.slurm_ngpus,
            gpus_per_node=args.slurm_ngpus,
            tasks_per_node=args.slurm_ngpus,
            cpus_per_task=2 if not args.slurm_ncpus else args.slurm_ncpus,
            nodes=args.slurm_nnodes,
            timeout_min=2800,
            slurm_partition=args.slurm_partition,
            slurm_exclude=args.slurm_exclude,
        )

        if args.slurm_nodelist:
            executor.update_parameters(
                slurm_additional_parameters={"nodelist": f"{args.slurm_nodelist}"}
            )

        executor.update_parameters(name="VAE_DG")
        trainer = SLURM_Trainer(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id} for VAE_DG")

        with open_dict(args):
            args.job_id = job.job_id

    else:
        with open_dict(args):
            args.job_id = None
        train(args)


def train(args):
    batch_size = args.train.batch_size
    image_size = args.train.image_size
    lr = args.train.lr
    nepoch = args.train.nepoch
    early_stop_patience = args.train.early_stop_patience
    scheduler_patience = args.train.scheduler_patience

    Path(args.model_out_path).mkdir(exist_ok=True, parents=True)
    Path(args.results_out_path).mkdir(exist_ok=True, parents=True)

    device = torch.device("cuda")

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
        deep_model=args.vae.deep_model,
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
    train_loss_log = []
    epoch_val_loss = []
    epoch_train_loss = []
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

            train_loss_log.append(loss.item())
            vae_net.zero_grad()
            loss.backward()
            optimizer.step()

            pbar_batches.update(1)
            pbar_batches.set_postfix(loss=loss.item())
        pbar_batches.close()

        epoch_train_loss.append(np.mean(train_loss_log))

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

        epoch_val_loss.append(np.mean(val_loss_log))
        scheduler.step(np.mean(val_loss_log))

        if np.mean(val_loss_log) < lowest_val_loss:
            Utils.remove_old_models(Path(args.model_out_path))
            torch.save(
                {
                    "epoch": epoch,
                    "val_loss_log": np.mean(val_loss_log),
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
            ic(f"\nEarly stop count: {early_stop_count}")

        if early_stop_count >= early_stop_patience:
            ic(f"\nEarly stop patience reached, stop the training")
            break

        pbar_epochs.update(1)
        pbar_epochs.set_postfix(mean_loss=np.mean(val_loss_log))
        val_loss_log.clear()
        train_loss_log.clear()
    pbar_epochs.close()

    # TEST
    # Carica il file salvato
    checkpoint = torch.load(
        f"{args.model_out_path}/val_loss_{lowest_val_loss:.4f}.pt", weights_only=False
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

    loss_df = pd.DataFrame({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss})
    loss_df.to_csv(f"{args.model_out_path}/train_val_loss.csv", index=False)


if __name__ == "__main__":
    main()
