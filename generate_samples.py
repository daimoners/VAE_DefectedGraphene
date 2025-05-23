try:
    import torch
    import torchvision.utils as vutils
    from tqdm import tqdm
    from lib.networks import VAE, get_resnet_model
    from pathlib import Path
    from lib.utils import (
        Utils,
        convert_isolated_black_pixels,
        new_new_draw_graphene_lattice,
        conta_pixel_neri,
    )
    import hydra
    import cv2
    from lib.networks import padding_image
    import numpy as np
    import submitit
    from omegaconf import open_dict
    from icecream import ic
    import os
    from PIL import Image
    from torchvision import transforms
    import time

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        generate_multiple_images(self.args)
        # generate_single_image(self.args)


def get_discriminator_checkpoint(models_dir: Path) -> str:
    if not models_dir.is_dir():
        raise Exception(f"{models_dir} is not a directory!")

    ckpts = [
        f
        for f in models_dir.iterdir()
        if (f.suffix.lower() == ".pt" and f.stem.lower().startswith("best"))
    ]

    return str(ckpts[0])


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

        executor.update_parameters(name="GEN_DG")
        trainer = SLURM_Trainer(args)
        job = executor.submit(trainer)
        print(f"Submitted job_id: {job.job_id} for GEN_DG")

        with open_dict(args):
            args.job_id = job.job_id

    else:
        with open_dict(args):
            args.job_id = None
        # generate_single_image(args)
        generate_multiple_images(args)


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
    package_path = Path(args.package_path)

    device = torch.device("cuda")

    img = load_image(
        package_path.joinpath("data", "perfect_graphene", "opt_graphene.png"),
        resolution=args.train.image_size,
    )
    img = img.to(device)

    # Create VAE network
    vae_atoms = VAE(
        channel_in=1,
        ch=args.vae.ch,
        blocks=tuple(args.vae.blocks),
        latent_channels=args.vae.latent_channels,
        deep_model=args.vae.deep_model,
    ).to(device)

    checkpoint = torch.load(
        get_checkpoint(Path(args.model_out_path)), weights_only=True
    )

    vae_atoms.load_state_dict(checkpoint["model_state_dict"])
    vae_atoms.eval()

    _, mu, logvar = vae_atoms(img)

    std = torch.exp(0.5 * logvar)

    epsilon = torch.randn_like(std)
    z = mu + std * epsilon
    z = z + 0.75 * torch.randn_like(z)

    recon_img = vae_atoms.decoder(z)
    recon_img = (recon_img >= 0.5).float()

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
    new_new_draw_graphene_lattice(f"{args.results_out_path}/single_generated.png")

    convert_isolated_black_pixels(
        f"{args.results_out_path}/single_generated_with_bonds.png",
        f"{args.results_out_path}/final.png",
    )


@hydra.main(version_base="1.2", config_path="config", config_name="cfg")
def generate_multiple_images(args):
    Path(args.results_out_path).joinpath("generated_dataset").mkdir(
        exist_ok=True, parents=True
    )
    package_path = Path(args.package_path)

    n_images = 5000

    device = torch.device("cuda")

    img = load_image(
        package_path.joinpath("data", "perfect_graphene", "opt_graphene.png"),
        resolution=args.train.image_size,
    )
    img = img.to(device)

    # Create VAE network
    vae_atoms = VAE(
        channel_in=1,
        ch=args.vae.ch,
        blocks=tuple(args.vae.blocks),
        latent_channels=args.vae.latent_channels,
        deep_model=args.vae.deep_model,
    ).to(device)

    checkpoint = torch.load(
        get_checkpoint(Path(args.model_out_path)), weights_only=False
    )

    vae_atoms.load_state_dict(checkpoint["model_state_dict"])
    vae_atoms.eval()

    _, mu, logvar = vae_atoms(img)

    resnet_ckpt = torch.load(
        get_discriminator_checkpoint(
            package_path.joinpath("data", "discriminator", "model")
        )
    )
    resnet = get_resnet_model()
    resnet.load_state_dict(resnet_ckpt["model_state_dict"])
    resnet.eval()

    CLASSES = resnet_ckpt["class_names"]

    data_augmentation_test = transforms.Compose(
        [
            transforms.Resize(args.train.image_size),
            transforms.Grayscale(num_output_channels=1),  # Converti in grayscale
            transforms.ToTensor(),
        ]
    )

    pbar = tqdm(total=n_images)
    i = 0
    start = time.time()
    while i < n_images:
        # Reparametrization trick per campionare dallo spazio latente
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        z = mu + std * epsilon
        z = z + 0.75 * torch.randn_like(z)

        recon_img = vae_atoms.decoder(z)
        recon_img = (recon_img >= 0.5).float()

        vutils.save_image(
            recon_img.cpu(),
            f"{args.results_out_path}/generated_dataset/generated_{i}.png",
            normalize=True,
        )

        atoms = conta_pixel_neri(
            Path(f"{args.results_out_path}/generated_dataset/generated_{i}.png")
        )
        new_new_draw_graphene_lattice(
            f"{args.results_out_path}/generated_dataset/generated_{i}.png"
        )

        count = convert_isolated_black_pixels(
            f"{args.results_out_path}/generated_dataset/generated_{i}_with_bonds.png",
            f"{args.results_out_path}/generated_dataset/generated_{i}.png",
        )
        atoms -= count
        os.remove(
            f"{args.results_out_path}/generated_dataset/generated_{i}_with_bonds.png"
        )

        img = Image.open(
            str(f"{args.results_out_path}/generated_dataset/generated_{i}.png")
        )
        img_pytorch = data_augmentation_test(img)
        img_pytorch = torch.unsqueeze(img_pytorch, dim=0)

        with torch.no_grad():
            output = np.squeeze(resnet(img_pytorch))
            # class_prediction = int(torch.round(torch.sigmoid(output)))

            if torch.sigmoid(output) >= 0.90:
                class_prediction = 1
            else:
                class_prediction = 0

            if CLASSES[class_prediction] == "broken" or atoms >= 512:
                ic(f"Broken predicted label for generated_{i}.png")
                os.remove(
                    f"{args.results_out_path}/generated_dataset/generated_{i}.png"
                )
            elif CLASSES[class_prediction] == "ok":
                np.savetxt(
                    f"{args.results_out_path}/generated_dataset/generated_{i}.txt",
                    np.array([atoms]),
                )
                i += 1
                pbar.update(1)
            else:
                raise Exception(f"Wrong classes for: {CLASSES[class_prediction]}")

    pbar.close()
    end = time.time()
    print(f"DG flakes generation complete ({(end - start) / 60:.2f} minutes)")


if __name__ == "__main__":
    main()
