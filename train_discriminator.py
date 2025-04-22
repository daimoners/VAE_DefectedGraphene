try:
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    import torch
    from torchvision import transforms, datasets
    import torch.nn as nn
    import torch.optim as optim
    from tqdm import tqdm
    import time
    import os
    import submitit
    import hydra
    from icecream import ic
    from omegaconf import open_dict
    import torchvision.models as models
    import pandas as pd
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


class SLURM_Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        discriminator(self.args)


IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
BINARY = True
CLASSES = ["broken", "ok"]
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = (240, 240)
BATCH_SIZE = 32
EPOCHS = 10  # 300
LEARNING_RATE = 0.001
NUM_WORKERS = 6

daimon_colors = ["#F0741E", "#276CB3"]
custom_cmap = LinearSegmentedColormap.from_list(
    "custom_cmap", ["#ffff", daimon_colors[1]]
)


def get_best_model_path(models_path: Path):
    models = [
        f
        for f in models_path.iterdir()
        if (f.suffix.lower() == ".pt" and f.stem.startswith("best_model_epoch"))
    ]
    if len(models) > 1:
        raise Exception(f"Error, {len(models)} best models found!")
    else:
        return models[0]


def delete_old_models(models_path: Path):
    if not models_path.is_dir():
        raise FileNotFoundError

    models = [
        f
        for f in models_path.iterdir()
        if (f.suffix.lower() == ".pt" and f.stem.startswith("best_model_epoch"))
    ]

    for model in models:
        os.remove(str(model))


def get_resnet_model(in_channels=1, out_channels=1):
    resnet18 = models.resnet18()

    resnet18.conv1 = nn.Conv2d(
        in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    resnet18.fc = nn.Sequential(
        nn.Linear(resnet18.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, out_channels),
    )

    return resnet18


class MySimpleClassifier(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(MySimpleClassifier, self).__init__()

        self.conv128 = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm2d(self.conv128.out_channels)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.conv256_1 = nn.Conv2d(
            in_channels=self.conv128.out_channels, out_channels=256, kernel_size=3
        )
        self.conv256_2 = nn.Conv2d(
            in_channels=self.conv256_1.out_channels, out_channels=256, kernel_size=3
        )
        self.batchnorm2 = nn.BatchNorm2d(self.conv256_2.out_channels)
        self.conv512_1 = nn.Conv2d(
            in_channels=self.conv256_2.out_channels, out_channels=512, kernel_size=3
        )
        self.conv512_2 = nn.Conv2d(
            in_channels=self.conv512_1.out_channels, out_channels=512, kernel_size=3
        )
        self.batchnorm3 = nn.BatchNorm2d(self.conv512_2.out_channels)

        self.flatten = nn.Flatten()
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(self.find_dimenstion(input_shape), 1024)
        self.batchnorm4 = nn.BatchNorm1d(self.fc1.out_features)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, num_classes)

    def forward(self, input):
        x = self.relu(self.conv128(input))
        x = self.batchnorm1(x)
        x = self.max_pool(x)
        x = self.relu(self.conv256_1(x))
        x = self.relu(self.conv256_2(x))
        x = self.batchnorm2(x)
        x = self.max_pool(x)
        x = self.relu(self.conv512_1(x))
        x = self.relu(self.conv512_2(x))
        x = self.batchnorm3(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        x = self.relu(self.fc2(x))

        return self.fc3(x)

    def find_dimenstion(self, input_shape):
        x = torch.rand(1, input_shape[2], input_shape[0], input_shape[1])

        x = self.relu(self.conv128(x))
        x = self.batchnorm1(x)
        x = self.max_pool(x)
        x = self.relu(self.conv256_1(x))
        x = self.relu(self.conv256_2(x))
        x = self.batchnorm2(x)
        x = self.max_pool(x)
        x = self.relu(self.conv512_1(x))
        x = self.relu(self.conv512_2(x))
        x = self.batchnorm3(x)
        x = self.max_pool(x)
        x = self.flatten(x)

        return x.size()[1]


def save_model(model, loss_function, optimizer, epoch, val_epoch_loss, savepath):
    savepath.mkdir(exist_ok=True, parents=True)

    delete_old_models(savepath)

    print("Saving model...")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_function,
            "class_names": CLASSES,
        },
        str(
            savepath.joinpath(f"best_model_epoch_{epoch}_vloss_{val_epoch_loss:.4f}.pt")
        ),
    )


def compute_accuracy(predictions, labels, type):
    if type == "binary":
        predictions_class = torch.round(torch.sigmoid(predictions))
    elif type == "multiclass":
        predictions_class = torch.argmax(predictions, dim=1)

    sum_of_matches = (predictions_class == labels).sum().float()

    accuracy = torch.round((sum_of_matches / labels.shape[0]) * 100)

    return accuracy


def training_step(model, train_dataloader, loss_function, optimizer, DEVICE_NAME):
    print("Training...")
    model.train()
    train_step_loss = 0.0
    train_step_accuracy = 0.0
    for i, data in tqdm(
        enumerate(train_dataloader),
        total=int(len(train_dataloader.dataset) / train_dataloader.batch_size),
    ):
        if BINARY:
            images, labels = (
                data[0].to(DEVICE_NAME),
                data[1].to(DEVICE_NAME, torch.float32),
            )
        else:
            images, labels = data[0].to(DEVICE_NAME), data[1].to(DEVICE_NAME)

        optimizer.zero_grad()

        outputs = model(images)

        loss = (
            loss_function(outputs, torch.unsqueeze(labels, dim=1))
            if BINARY
            else loss_function(outputs, labels)
        )

        accuracy = (
            compute_accuracy(outputs, torch.unsqueeze(labels, dim=1), type="binary")
            if BINARY
            else compute_accuracy(outputs, labels, type="multiclass")
        )

        train_step_loss += loss.item()
        train_step_accuracy += accuracy.item()

        loss.backward()
        optimizer.step()

    train_loss = train_step_loss / len(train_dataloader)
    train_acc = train_step_accuracy / len(train_dataloader)

    print(f"Train Loss: {train_loss:.5f}, Train Acc: {train_acc:.2f}%\n")

    return train_loss


def validation_step(model, val_dataloader, loss_function, optimizer, DEVICE_NAME):
    print("Validating...")
    model.eval()
    val_step_loss = 0.0
    val_step_accuracy = 0.0
    with torch.no_grad():
        for i, data in tqdm(
            enumerate(val_dataloader),
            total=int(len(val_dataloader.dataset) / val_dataloader.batch_size),
        ):
            if BINARY:
                images, labels = (
                    data[0].to(DEVICE_NAME),
                    data[1].to(DEVICE_NAME, torch.float32),
                )
            else:
                images, labels = data[0].to(DEVICE_NAME), data[1].to(DEVICE_NAME)

            outputs = model(images)

            loss = (
                loss_function(outputs, torch.unsqueeze(labels, dim=1))
                if BINARY
                else loss_function(outputs, labels)
            )
            accuracy = (
                compute_accuracy(outputs, torch.unsqueeze(labels, dim=1), type="binary")
                if BINARY
                else compute_accuracy(outputs, labels, type="multiclass")
            )

            val_step_loss += loss.item()
            val_step_accuracy += accuracy.item()

        val_loss = val_step_loss / len(val_dataloader)
        val_acc = val_step_accuracy / len(val_dataloader)

        print(f"Val Loss: {val_loss:.5f}, Val Acc: {val_acc:.2f}%")

        return val_loss


def train(
    model,
    train_dataloader,
    val_dataloader,
    loss_function,
    optimizer,
    DEVICE_NAME,
    package_path,
):
    lowest_val_loss = float("inf")
    start = time.time()

    train_loss_log = []
    val_loss_log = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1} of {EPOCHS}")
        train_epoch_loss = training_step(
            model, train_dataloader, loss_function, optimizer, DEVICE_NAME
        )
        val_epoch_loss = validation_step(
            model, val_dataloader, loss_function, optimizer, DEVICE_NAME
        )

        if val_epoch_loss < lowest_val_loss:
            save_model(
                model,
                loss_function,
                optimizer,
                epoch,
                val_epoch_loss,
                package_path.joinpath("model"),
            )
            lowest_val_loss = val_epoch_loss
            ic(lowest_val_loss)

        print(f"Elapsed time: {(time.time() - start) / 60:.3f} minutes\n")

        train_loss_log.append(train_epoch_loss)
        val_loss_log.append(val_epoch_loss)
    end = time.time()
    loss_df = pd.DataFrame({"train_loss": train_loss_log, "val_loss": val_loss_log})
    loss_df.to_csv(
        str(package_path.joinpath("discriminator_train_val_loss.csv")), index=False
    )
    print(f"{(end - start) / 60:.3f} minutes")


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
        discriminator(args)


def discriminator(args):
    package_path = Path(args.package_path).joinpath("data", "discriminator")
    model_path = package_path.joinpath("model")
    model_path.mkdir(exist_ok=True, parents=True)
    results_path = package_path.joinpath("results")
    results_path.mkdir(exist_ok=True, parents=True)

    DEVICE_NAME = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device = {DEVICE_NAME}")

    data_augmentation_train = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    data_augmentation_val = data_augmentation_test = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ]
    )

    def create_pytorch_dataloader(train_path: Path, val_path: Path):
        train_folder = datasets.ImageFolder(str(train_path), data_augmentation_train)
        val_folder = datasets.ImageFolder(str(val_path), data_augmentation_val)

        train_dataloader = torch.utils.data.DataLoader(
            train_folder, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_folder, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )

        return train_dataloader, val_dataloader

    train_path = package_path.joinpath("dataset", "train")
    val_path = package_path.joinpath("dataset", "val")

    train_dataloader, val_dataloader = create_pytorch_dataloader(train_path, val_path)

    model = get_resnet_model()
    loss_function = nn.BCEWithLogitsLoss() if BINARY else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    model.to(DEVICE_NAME)
    train(
        model,
        train_dataloader,
        val_dataloader,
        loss_function,
        optimizer,
        DEVICE_NAME,
        package_path,
    )

    model = get_resnet_model()
    model.load_state_dict(
        torch.load(str(get_best_model_path(package_path.joinpath("model"))))[
            "model_state_dict"
        ]
    )
    model.eval()

    test_path = package_path.joinpath("dataset", "test")

    samples_broken = [
        x
        for x in test_path.joinpath("broken").iterdir()
        if x.suffix in IMAGE_EXTENSIONS
    ]
    samples_ok = [
        x for x in test_path.joinpath("ok").iterdir() if x.suffix in IMAGE_EXTENSIONS
    ]

    y_true = []
    y_pred = []

    labels_map = {"broken": 0, "ok": 1}

    for label, samples in [("broken", samples_broken), ("ok", samples_ok)]:
        for img_path in samples:
            img = Image.open(str(img_path))
            img_pytorch = data_augmentation_test(img)
            img_pytorch = torch.unsqueeze(img_pytorch, dim=0)

            with torch.no_grad():
                output = np.squeeze(model(img_pytorch))
                class_prediction = int(torch.round(torch.sigmoid(output)))

            y_true.append(labels_map[label])
            y_pred.append(class_prediction)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    np.savetxt(str(results_path.joinpath("cmatrix.txt")), cm)
    labels = ["Broken", "OK"]

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=custom_cmap,
        cbar=True,
        linewidths=0.5,
        linecolor="black",
        square=True,
        vmin=0,
        vmax=np.max(cm),
    )
    plt.xlabel("Predicted", fontsize=23)
    plt.ylabel("Actual", fontsize=23)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)

    ax.set_xticklabels(labels, fontsize=17)
    ax.set_yticklabels(labels, fontsize=17)

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(0.5)

    plt.tight_layout()
    plt.tick_params(axis="both", which="major")
    plt.savefig(
        str(results_path.joinpath("cmatrix.png")),
        dpi=300,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
