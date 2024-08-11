try:
    import torch
    import torchvision.utils as vutils
    from tqdm import tqdm
    from lib.networks import VAE
    from pathlib import Path
    from lib.utils import (
        Utils,
        convert_isolated_black_pixels,
        draw_graphene_lattice,
    )
    import hydra
    import cv2
    from lib.networks import padding_image
    import numpy as np
    import submitit
    from omegaconf import open_dict, OmegaConf
    from icecream import ic

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        generate_single_image(self.args)


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
        generate_single_image(args)


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

    device = torch.device("cuda")

    img = load_image(
        "/home/tommaso/git_workspace/VAE_DefectedGraphene/data/perfect_graphene/opt_graphene.png",
        resolution=args.train.image_size,
    )
    img = img.to(device)
    print(img.shape)

    # Create VAE network
    vae_atoms = VAE(
        channel_in=1,
        ch=args.vae.ch,
        blocks=tuple(args.vae.blocks),
        latent_channels=args.vae.latent_channels,
        deep_model=args.vae.deep_model,
    ).to(device)

    # Carica il file salvato
    checkpoint = torch.load(
        get_checkpoint(Path(args.model_out_path)), weights_only=True
    )
    # Ripristina lo stato del modello e dell'ottimizzatore
    vae_atoms.load_state_dict(checkpoint["model_state_dict"])
    vae_atoms.eval()

    _, mu, logvar = vae_atoms(img)

    # Reparametrization trick per campionare dallo spazio latente
    std = torch.exp(0.5 * logvar)

    # scale_factor = 1.5
    # epsilon = torch.randn_like(std) * scale_factor

    noise_factor = 0.8
    # mu_noisy = mu + noise_factor * torch.randn_like(mu)

    # z = mu_noisy + std * epsilon

    epsilon = torch.randn_like(std)
    z = mu + std * epsilon
    z = z + 0.9 * torch.randn_like(z)

    print(z.shape)
    print(torch.max(z))
    print(torch.min(z))

    recon_img = vae_atoms.decoder(z)
    recon_img = (recon_img >= 0.5).float()
    print(torch.max(recon_img))
    print(torch.min(recon_img))

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
    draw_graphene_lattice(f"{args.results_out_path}/single_generated.png")

    convert_isolated_black_pixels(
        f"{args.results_out_path}/single_generated_with_bonds.png",
        f"{args.results_out_path}/final.png",
    )


if __name__ == "__main__":
    main()
