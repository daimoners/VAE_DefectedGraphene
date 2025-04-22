try:
    from lib.utils import Utils
    from pathlib import Path
    import shutil

except Exception as e:
    print(f"Some module are missing from {__file__}: {e}\n")


def main():
    max_dim = [39.53476932, 34.27629786]
    xyz_files_path = Path("./data/xyz_files")
    images_path = Path("./data/images")
    Utils.from_xyz_to_png(
        xyz_files_path,
        images_path,
        max_dim=max_dim,
        multiplier=6,
    )
    Utils.split_images(
        images_path, dataset_path=images_path.parent.joinpath("training_dataset")
    )
    shutil.rmtree(images_path)


if __name__ == "__main__":
    main()
