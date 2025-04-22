try:
    import numpy as np
    import shutil
    from tqdm import tqdm
    import random
    from pathlib import Path
    from chemfiles import Trajectory
    from PIL import Image, ImageDraw
    import os
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import cv2
    from numba import njit
    from scipy.spatial import distance
    from lib.networks import MyDatasetPng, MyDatasetPngMixed

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
            dataset_path = images_atoms.parent.joinpath("training_dataset")

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


@njit
def process_image(pixels, width, height):
    count = 0
    # Scorriamo tutti i pixel dell'immagine
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # Controlla se il pixel attuale è nero
            if pixels[y, x] == 0:
                # Controlla i pixel adiacenti
                if (
                    pixels[y - 1, x] == 255
                    and pixels[y + 1, x] == 255
                    and pixels[y, x - 1] == 255
                    and pixels[y, x + 1] == 255
                    and pixels[y - 1, x - 1] == 255
                    and pixels[y - 1, x + 1] == 255
                    and pixels[y + 1, x - 1] == 255
                    and pixels[y + 1, x + 1] == 255
                ):
                    # Se tutti i pixel adiacenti sono bianchi, cambia il pixel attuale in bianco
                    count += 1
                    pixels[y, x] = 255

    return count


def convert_isolated_black_pixels(image_path, dpath):
    # Apri l'immagine e convertila in scala di grigi
    img = Image.open(image_path).convert("L")
    pixels = np.array(img)

    # Ottieni le dimensioni dell'immagine
    height, width = pixels.shape

    # Processa l'immagine con la funzione accelerata da Numba
    count = process_image(pixels, width, height)

    # Converti l'array modificato di nuovo in immagine
    new_img = Image.fromarray(pixels)

    new_img.save(dpath)

    return count


def draw_graphene_lattice(image_path, min_bond_length=7, max_bond_length=11):
    # Carica l'immagine
    binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    black_pixels = []
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 0:  # Se il valore del pixel è 0 (nero)
                black_pixels.append((x, y))

    # Converti i punti in un array numpy per calcolare le distanze
    points = np.array(black_pixels)

    # Traccia i legami tra i punti vicini
    indices = []
    for i, point in enumerate(points):
        point_i = points[i]
        for j in range(i + 1, len(points)):
            point_j = points[j]

            # Calcola la distanza euclidea tra i punti
            dist = distance.euclidean(point_i, point_j)

            # Se la distanza è inferiore o uguale al limite minimo
            if dist < min_bond_length:
                # Imposta il raggio iniziale
                radius = max_bond_length

                # Riduci il raggio finché i vicini sono uguali
                while True:
                    neighbors_i = find_neighbors(points=points, idx=i, radius=radius)
                    neighbors_j = find_neighbors(points=points, idx=j, radius=radius)

                    # Se i vicini non sono uguali, esci dal ciclo
                    if neighbors_i != neighbors_j:
                        break

                    # Riduci il raggio
                    radius -= 1

                    # Se il raggio diventa troppo piccolo, esci dal ciclo
                    if radius < 0:
                        radius = 0
                        break

                # Se il numero di vicini per il punto i è maggiore, aggiungi j agli indici
                if neighbors_i > neighbors_j:
                    indices.append(j)
                elif neighbors_j > neighbors_i:
                    indices.append(i)

    points = np.delete(points, indices, axis=0)

    # Crea una copia dell'immagine per disegnare i legami
    img_with_bonds = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Traccia i legami tra i punti vicini
    for i, point in enumerate(points):
        for j in range(i + 1, len(points)):
            if (
                min_bond_length
                <= distance.euclidean(points[i], points[j])
                <= max_bond_length
            ):
                cv2.line(
                    img_with_bonds, tuple(points[i]), tuple(points[j]), (0, 0, 0), 1
                )

    # Salva l'immagine risultante
    output_path = image_path.replace(".png", "_with_bonds.png")
    cv2.imwrite(output_path, img_with_bonds)


def new_draw_graphene_lattice(image_path, min_bond_length=7, max_bond_length=11):
    # Carica l'immagine
    binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    black_pixels = []
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 0:  # Se il valore del pixel è 0 (nero)
                black_pixels.append((x, y))

    # Converti i punti in un array numpy per calcolare le distanze
    points = np.array(black_pixels)

    # Traccia i legami tra i punti vicini
    indices = []
    for i, point in enumerate(points):
        point_i = points[i]
        for j in range(i + 1, len(points)):
            point_j = points[j]

            # Calcola la distanza euclidea tra i punti
            dist = distance.euclidean(point_i, point_j)

            # Se la distanza è inferiore o uguale al limite minimo
            if dist < min_bond_length:
                # Imposta il raggio iniziale
                radius = max_bond_length

                # Riduci il raggio finché i vicini sono uguali
                while True:
                    neighbors_i = find_neighbors(points=points, idx=i, radius=radius)
                    neighbors_j = find_neighbors(points=points, idx=j, radius=radius)

                    # Se i vicini non sono uguali, esci dal ciclo
                    if neighbors_i != neighbors_j:
                        break

                    # Riduci il raggio
                    radius -= 1

                    # Se il raggio diventa troppo piccolo, esci dal ciclo
                    if radius < 0:
                        radius = 0
                        break

                # Se il numero di vicini per il punto i è maggiore, aggiungi j agli indici
                if neighbors_i > neighbors_j:
                    indices.append(j)
                elif neighbors_j > neighbors_i:
                    indices.append(i)

    points = np.delete(points, indices, axis=0)

    # Dizionario per contare il numero di legami per ogni punto
    bonds_count = {i: 0 for i in range(len(points))}

    # Lista per memorizzare i legami (collegamenti validi tra i punti)
    valid_bonds = []

    # Traccia i legami tra i punti vicini
    for i, point_i in enumerate(points):
        for j in range(i + 1, len(points)):
            point_j = points[j]

            # Calcola la distanza euclidea tra i punti
            dist = distance.euclidean(point_i, point_j)

            # Se la distanza è compresa tra il limite minimo e massimo
            if min_bond_length <= dist <= max_bond_length:
                # Aggiungi il legame se valido
                bonds_count[i] += 1
                bonds_count[j] += 1
                valid_bonds.append((i, j))
                if (point_i[0] <= 8 or point_i[0] >= 230) or (
                    point_j[0] <= 8 or point_j[0] >= 230
                ):
                    bonds_count[i] += 1
                    bonds_count[j] += 1

    # Crea una copia dell'immagine per disegnare i legami
    img_with_bonds = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Disegna i legami tra i punti che hanno almeno 2 legami
    for i, j in valid_bonds:
        if bonds_count[i] >= 2 and bonds_count[j] >= 2:
            cv2.line(img_with_bonds, tuple(points[i]), tuple(points[j]), (0, 0, 0), 1)

    # Salva l'immagine risultante
    output_path = image_path.replace(".png", "_with_bonds.png")
    cv2.imwrite(output_path, img_with_bonds)


def new_new_draw_graphene_lattice(
    image_path, dpath=None, min_bond_length=7, max_bond_length=11
):
    # Carica l'immagine
    binary = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    black_pixels = []
    for y in range(binary.shape[0]):
        for x in range(binary.shape[1]):
            if binary[y, x] == 0:  # Se il valore del pixel è 0 (nero)
                black_pixels.append((x, y))

    # Converti i punti in un array numpy per calcolare le distanze
    points = np.array(black_pixels)

    # Traccia i legami tra i punti vicini
    indices = []
    for i, point in enumerate(points):
        point_i = points[i]
        for j in range(i + 1, len(points)):
            point_j = points[j]

            # Calcola la distanza euclidea tra i punti
            dist = distance.euclidean(point_i, point_j)

            # Se la distanza è inferiore o uguale al limite minimo
            if dist < min_bond_length:
                # Imposta il raggio iniziale
                radius = max_bond_length

                # Riduci il raggio finché i vicini sono uguali
                while True:
                    neighbors_i = find_neighbors(points=points, idx=i, radius=radius)
                    neighbors_j = find_neighbors(points=points, idx=j, radius=radius)

                    # Se i vicini non sono uguali, esci dal ciclo
                    if neighbors_i != neighbors_j:
                        break

                    # Riduci il raggio
                    radius -= 1

                    # Se il raggio diventa troppo piccolo, esci dal ciclo
                    if radius < 0:
                        radius = 0
                        break

                # Se il numero di vicini per il punto i è maggiore, aggiungi j agli indici
                if neighbors_i > neighbors_j:
                    indices.append(j)
                elif neighbors_j > neighbors_i:
                    indices.append(i)

    points = np.delete(points, indices, axis=0)

    # Dizionario per contare il numero di legami per ogni punto
    bonds_count = {i: 0 for i in range(len(points))}

    # Lista per memorizzare i legami (collegamenti validi tra i punti)
    valid_bonds = []

    neighbors = {i: [] for i in range(len(points))}

    # Traccia i legami tra i punti vicini
    for i, point_i in enumerate(points):
        for j in range(i + 1, len(points)):
            point_j = points[j]

            # Calcola la distanza euclidea tra i punti
            dist = distance.euclidean(point_i, point_j)

            # Se la distanza è compresa tra il limite minimo e massimo
            if min_bond_length <= dist <= max_bond_length:
                # Aggiungi il legame se valido
                bonds_count[i] += 1
                bonds_count[j] += 1
                valid_bonds.append((i, j))
                neighbors[i].append(j)
                neighbors[j].append(i)
                if (point_i[0] <= 8 or point_i[0] >= 230) or (
                    point_j[0] <= 8 or point_j[0] >= 230
                ):
                    bonds_count[i] = 100
                    bonds_count[j] = 100
    # test_points = {i: point for i, point in enumerate(points)}
    # ic(test_points)
    # ic(neighbors)
    # Crea una copia dell'immagine per disegnare i legami
    img_with_bonds = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # Disegna i legami tra i punti che hanno almeno 2 legami
    for i, j in valid_bonds:
        if len(neighbors[i]) >= 2 and len(neighbors[j]) >= 2:
            draw = True
            for n in neighbors[i]:
                if len(neighbors[n]) < 2 and bonds_count[n] != 100:
                    draw = False
            for n in neighbors[j]:
                if len(neighbors[n]) < 2 and bonds_count[n] != 100:
                    draw = False

            if draw or (len(neighbors[i]) >= 3 and len(neighbors[j]) >= 3):
                cv2.line(
                    img_with_bonds, tuple(points[i]), tuple(points[j]), (0, 0, 0), 1
                )

        if bonds_count[i] == 100 or bonds_count[j] == 100:
            cv2.line(img_with_bonds, tuple(points[i]), tuple(points[j]), (0, 0, 0), 1)

    # Salva l'immagine risultante
    if dpath is None:
        output_path = image_path.replace(".png", "_with_bonds.png")
    else:
        output_path = dpath
    cv2.imwrite(output_path, img_with_bonds)


def find_average_distance(points, idx, radius):
    # Lista per memorizzare le distanze ai vicini
    p1 = points[idx]
    distances = []

    # Itera su tutti gli altri punti per trovare i vicini
    for j, p2 in enumerate(points):
        if j != idx:
            # Calcola la distanza euclidea tra p1 e p2
            distance = np.linalg.norm(p1 - p2)

            # Se la distanza è entro il raggio specificato, aggiungi alla lista
            if distance <= radius:
                distances.append(distance)

    # Calcola la distanza media, se ci sono vicini
    if distances:
        average_distance = np.mean(distances)
    else:
        average_distance = 0  # Se non ci sono vicini, la distanza media è 0

    return average_distance


def find_neighbors(points, idx, radius):
    # Lista per memorizzare le distanze ai vicini
    p1 = points[idx]
    neighbors = 0

    # Itera su tutti gli altri punti per trovare i vicini
    for j, p2 in enumerate(points):
        if j != idx:
            # Calcola la distanza euclidea tra p1 e p2
            distance = np.linalg.norm(p1 - p2)

            # Se la distanza è entro il raggio specificato, aggiungi alla lista
            if distance <= radius:
                neighbors += 1

    return neighbors


def conta_pixel_neri(image_path):
    # Carica l'immagine
    if isinstance(image_path, Path):
        img = Image.open(image_path).convert("L")  # Converti in scala di grigi
    else:
        img = image_path

    # Converti l'immagine in un array NumPy
    img_array = np.array(img)

    # Conta i pixel neri (valore 0 in scala di grigi)
    num_pixel_neri = np.sum(img_array == 0)

    return num_pixel_neri


def inverti_maschera_binaria(image_path):
    # Carica l'immagine e convertila in scala di grigi
    img = Image.open(image_path).convert("L")

    # Converti l'immagine in un array NumPy
    img_array = np.array(img)

    # Inverti i colori (da 0 a 255 e viceversa)
    img_inverted_array = 255 - img_array

    # Converti l'array invertito in immagine
    img_inverted = Image.fromarray(img_inverted_array)

    return img_inverted


if __name__ == "__main__":
    pass
