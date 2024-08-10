try:
    import numpy as np
    import shutil
    from tqdm import tqdm
    import random
    from PIL import Image
    from pathlib import Path
    from chemfiles import Trajectory
    from PIL import Image, ImageDraw
    from lib.networks import MyDatasetPng, MyDatasetPngMixed
    import os
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import cv2

except Exception as e:
    print("Some module are missing {}".format(e))


class Utils:
    IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")

    @staticmethod
    def generate_bonds_png(
        spath: Path,
        dpath: Path,
        max_dim: list,
        multiplier: int = 3,
    ):

        with Trajectory(str(spath)) as trajectory:
            mol = trajectory.read()

        resolution = round(
            multiplier * (5 + np.max([np.abs(max_dim[0]), np.abs(max_dim[1])]))
        )

        B = Image.new("RGB", (resolution, resolution))
        B_ = ImageDraw.Draw(B)

        mol.guess_bonds()
        if mol.topology.bonds_count() == 0:
            print(f"No bonds guessed for {spath.stem}\n")
        bonds = mol.topology.bonds

        for i in range(len(bonds)):
            x_1 = int(round(mol.positions[bonds[i][0]][0] * multiplier))
            y_1 = int(round(mol.positions[bonds[i][0]][1] * multiplier))
            x_2 = int(round(mol.positions[bonds[i][1]][0] * multiplier))
            y_2 = int(round(mol.positions[bonds[i][1]][1] * multiplier))
            line = [(x_1, y_1), (x_2, y_2)]
            first_atom = mol.atoms[bonds[i][0]].name
            second_atom = mol.atoms[bonds[i][1]].name
            color = Utils.find_bound_type(first_atom, second_atom)
            B_.line(line, fill=color, width=0)

        B = Utils.crop_image(B)
        B.save(str(dpath.joinpath(f"{spath.stem}.png")))

    def generate_atoms_png(
        spath: Path,
        dpath: Path,
        max_dim: list,
        multiplier: int = 3,
    ):

        with Trajectory(str(spath)) as trajectory:
            mol = trajectory.read()

        resolution = round(
            multiplier * (5 + np.max([np.abs(max_dim[0]), np.abs(max_dim[1])]))
        )

        B = Image.new("RGB", (resolution, resolution))
        B_ = ImageDraw.Draw(B)

        mol.guess_bonds()
        if mol.topology.bonds_count() == 0:
            print(f"No bonds guessed for {spath.stem}\n")
        bonds = mol.topology.bonds

        points = []

        for i in range(len(bonds)):
            x_1 = int(round(mol.positions[bonds[i][0]][0] * multiplier))
            y_1 = int(round(mol.positions[bonds[i][0]][1] * multiplier))
            x_2 = int(round(mol.positions[bonds[i][1]][0] * multiplier))
            y_2 = int(round(mol.positions[bonds[i][1]][1] * multiplier))
            first_atom = mol.atoms[bonds[i][0]].name
            second_atom = mol.atoms[bonds[i][1]].name
            color = Utils.find_bound_type(first_atom, second_atom)
            points.append((x_1, y_1))
            points.append((x_2, y_2))

        for point in points:
            B_.point(point, fill=color)

        B = Utils.crop_image(B)
        B.save(str(dpath.joinpath(f"{spath.stem}.png")))

    @staticmethod
    def find_bound_type(first_atom: str, second_atom: str) -> str:
        if (first_atom == "C" and second_atom == "C") or (
            second_atom == "C" and first_atom == "C"
        ):
            return "white"
        elif (first_atom == "C" and second_atom == "O") or (
            second_atom == "O" and first_atom == "C"
        ):
            return "blue"
        elif (first_atom == "O" and second_atom == "H") or (
            second_atom == "H" and first_atom == "O"
        ):
            return "red"
        elif (first_atom == "C" and second_atom == "H") or (
            second_atom == "H" and first_atom == "C"
        ):
            return "yellow"

    @staticmethod
    def crop_image(image: Image, name: str = None, dpath: Path = None) -> Image:

        image_data = np.asarray(image)
        if len(image_data.shape) == 2:
            image_data_bw = image_data
        else:
            image_data_bw = image_data.max(axis=2)
        non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
        non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
        cropBox = (
            min(non_empty_rows),
            max(non_empty_rows),
            min(non_empty_columns),
            max(non_empty_columns),
        )

        if len(image_data.shape) == 2:
            image_data_new = image_data[
                cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1
            ]
        else:
            image_data_new = image_data[
                cropBox[0] : cropBox[1] + 1, cropBox[2] : cropBox[3] + 1, :
            ]

        new_image = Image.fromarray(image_data_new)
        if dpath is not None:
            new_image.save(dpath.joinpath(name))

        return new_image

    @staticmethod
    def from_xyz_to_png(
        spath: Path,
        dpath: Path,
        max_dim: list,
        items: int = None,
        multiplier: int = 6,
    ):
        if dpath.is_dir():
            print(f"WARNING: the directory {dpath} already exists!")
            return
        else:
            dpath.mkdir(exist_ok=True, parents=True)

        files = [f for f in spath.iterdir() if f.suffix.lower() == ".xyz"]
        if items is None:
            items = len(files)

        pbar = tqdm(total=len(files) if items > len(files) else items)
        for i, file in enumerate(files):
            if i >= items:
                break
            # Utils.generate_bonds_png(file, dpath, max_dim, multiplier)
            Utils.generate_atoms_png(file, dpath, max_dim, multiplier)
            pbar.update(1)
        pbar.close()

    @staticmethod
    def grayscale_to_rgb(grayscale_tensor):
        # grayscale_tensor ha dimensioni [batch_size, 1, height, width]
        # La funzione repeat replica il canale grayscale per 3 volte lungo la dimensione dei canali
        rgb_tensor = grayscale_tensor.repeat(1, 3, 1, 1)
        return rgb_tensor

    @staticmethod
    def remove_old_models(models_path: Path):
        models = [f for f in models_path.iterdir() if f.suffix.lower() == ".pt"]
        for model in models:
            os.remove(model)

    @staticmethod
    def split_images(
        images_path: Path,
        dataset_path: Path = None,
        split_ratio: list = [0.7, 0.15, 0.15],
    ):
        if not images_path.is_dir():
            raise Exception(f"{images_path} is not a directory!")

        if dataset_path is None:
            dataset_path = images_path.parent.joinpath("training_dataset")

        train_path = dataset_path.joinpath("train")
        train_path.mkdir(exist_ok=True, parents=True)
        val_path = dataset_path.joinpath("val")
        val_path.mkdir(exist_ok=True, parents=True)
        test_path = dataset_path.joinpath("test")
        test_path.mkdir(exist_ok=True, parents=True)

        images = [
            f
            for f in images_path.iterdir()
            if f.suffix.lower() in Utils.IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        # Calculate the split indices
        length = len(images)
        train_split = int(split_ratio[0] * length)
        val_split = int((split_ratio[0] + split_ratio[1]) * length)

        # Create the three sublists
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for image_path in tqdm(train_images):
            shutil.copy(image_path, train_path.joinpath(image_path.name))

        for image_path in tqdm(val_images):
            shutil.copy(image_path, val_path.joinpath(image_path.name))

        for image_path in tqdm(test_images):
            shutil.copy(image_path, test_path.joinpath(image_path.name))

    @staticmethod
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
            f
            for f in train_path.iterdir()
            if (f.suffix.lower() in Utils.IMAGE_EXTENSIONS and "_B" not in f.stem)
        ]
        test_paths = [
            f
            for f in test_path.iterdir()
            if (f.suffix.lower() in Utils.IMAGE_EXTENSIONS and "_B" not in f.stem)
        ]
        val_paths = [
            f
            for f in val_path.iterdir()
            if (f.suffix.lower() in Utils.IMAGE_EXTENSIONS and "_B" not in f.stem)
        ]

        train_data = MyDatasetPng(
            train_paths,
            resolution=resolution,  # transforms=data_transform
        )
        test_data = MyDatasetPng(
            test_paths,
            resolution=resolution,  # transforms=data_transform
        )
        val_data = MyDatasetPng(
            val_paths,
            resolution=resolution,  # transforms=data_transform
        )

        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return trainloader, testloader, valloader

    @staticmethod
    def get_mixed_dataloaders(dataset_path: Path, batch_size: int, resolution: int):
        if not dataset_path.is_dir():
            raise Exception(f"{dataset_path} is not a directory!")

        train_path = dataset_path.joinpath("train")
        val_path = dataset_path.joinpath("val")
        test_path = dataset_path.joinpath("test")

        train_paths = [
            f
            for f in train_path.iterdir()
            if (f.suffix.lower() in Utils.IMAGE_EXTENSIONS and "_B" not in f.stem)
        ]
        test_paths = [
            f
            for f in test_path.iterdir()
            if (f.suffix.lower() in Utils.IMAGE_EXTENSIONS and "_B" not in f.stem)
        ]
        val_paths = [
            f
            for f in val_path.iterdir()
            if (f.suffix.lower() in Utils.IMAGE_EXTENSIONS and "_B" not in f.stem)
        ]

        train_data = MyDatasetPngMixed(
            train_paths,
            resolution=resolution,  # transforms=data_transform
        )
        test_data = MyDatasetPngMixed(
            test_paths,
            resolution=resolution,  # transforms=data_transform
        )
        val_data = MyDatasetPngMixed(
            val_paths,
            resolution=resolution,  # transforms=data_transform
        )

        trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return trainloader, testloader, valloader

    @staticmethod
    def apply_threshold(image_path: Path, threshold: int = 180, dpath: Path = None):
        if dpath is None:
            dpath = image_path.parent.joinpath(f"th_{image_path.name}")
        # Carica l'immagine in scala di grigi
        image = Image.open(image_path).convert("L")

        # Converti l'immagine in un array numpy
        image_array = np.array(image)

        # Applica il threshold: se il pixel è maggiore del threshold, diventa nero (0), altrimenti bianco (255)
        binary_image_array = np.where(image_array < threshold, 0, 255).astype(np.uint8)

        # Converti l'array di nuovo in un'immagine
        binary_image = Image.fromarray(binary_image_array)

        binary_image.save(dpath)

    @staticmethod
    def sharpened_image(image_path: Path, dpath: Path = None):
        if dpath is None:
            dpath = image_path.parent.joinpath(f"th_{image_path.name}")

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        # Definisci un kernel di sharpening
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        # Applica il filtro di sharpening
        sharpened_image = cv2.filter2D(image, -1, kernel)

        cv2.imwrite(str(dpath), sharpened_image)

    @staticmethod
    def generate_mixed_dataset(
        images_atoms: Path,
        images_bonds: Path,
        dataset_path: Path = None,
        split_ratio: list = [0.7, 0.15, 0.15],
    ):
        if not images_atoms.is_dir() or not images_bonds.is_dir():
            raise FileNotFoundError

        if dataset_path is None:
            dataset_path = images_path.parent.joinpath("training_dataset")

        train_path = dataset_path.joinpath("train")
        train_path.mkdir(exist_ok=True, parents=True)
        val_path = dataset_path.joinpath("val")
        val_path.mkdir(exist_ok=True, parents=True)
        test_path = dataset_path.joinpath("test")
        test_path.mkdir(exist_ok=True, parents=True)

        images = [
            f
            for f in images_atoms.iterdir()
            if f.suffix.lower() in Utils.IMAGE_EXTENSIONS
        ]

        random.shuffle(images)

        # Calculate the split indices
        length = len(images)
        train_split = int(split_ratio[0] * length)
        val_split = int((split_ratio[0] + split_ratio[1]) * length)

        # Create the three sublists
        train_images = images[:train_split]
        val_images = images[train_split:val_split]
        test_images = images[val_split:]

        for image_path in tqdm(train_images):
            shutil.copy(image_path, train_path.joinpath(image_path.name))
            shutil.copy(
                images_bonds.joinpath(image_path.name),
                train_path.joinpath(f"{image_path.stem}_B.png"),
            )

        for image_path in tqdm(val_images):
            shutil.copy(image_path, val_path.joinpath(image_path.name))
            shutil.copy(
                images_bonds.joinpath(image_path.name),
                val_path.joinpath(f"{image_path.stem}_B.png"),
            )

        for image_path in tqdm(test_images):
            shutil.copy(image_path, test_path.joinpath(image_path.name))
            shutil.copy(
                images_bonds.joinpath(image_path.name),
                test_path.joinpath(f"{image_path.stem}_B.png"),
            )


if __name__ == "__main__":
    # max_dim = [39.53476932, 34.27629786]
    # xyz_files_path = Path("../data/xyz_files")
    # images_path = Path("../data/tmp")
    # Utils.from_xyz_to_png(xyz_files_path, images_path, max_dim=max_dim, multiplier=6)
    # Utils.split_images(
    #     images_path, dataset_path=images_path.parent.joinpath("training_dataset_A")
    # )
    Utils.generate_mixed_dataset(
        images_atoms=Path("../data/images_240A"),
        images_bonds=Path("../data/images_240"),
        dataset_path=Path("../data/training_dataset_mixed"),
    )
    pass
