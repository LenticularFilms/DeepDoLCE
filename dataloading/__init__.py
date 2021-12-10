from .lenticular_square_patch_dataset import load_square_patch_dataset
from .lenticular_full_image_dataset import load_full_image_dataset
from .destriping_dataset import load_destriping_dataset

def get_dataloaders(args):
    if args.data == "lenticular_square_patch":
        return load_square_patch_dataset(args)
    if args.data == "lenticular_full_image":
        return load_full_image_dataset(args)
    if args.data == "destriping":
        return load_destriping_dataset(args)
    else:
        raise NotImplementedError
