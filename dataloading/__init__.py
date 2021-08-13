from .square_patch_dataset import load_square_patch_dataset
from .full_image_dataset import load_full_image_dataset
from .genome_dataset import load_genome_dataset
from .dolce_genome_dataset import load_dolce_plus_genome_dataset

def get_dataloaders(args):
    if args.data == "square_patch":
        return load_square_patch_dataset(args)
    if args.data == "full_image":
        return load_full_image_dataset(args)
    if args.data == "genome":
        return load_genome_dataset(args)
    if args.data == "dolce_plus_genome":
        return load_dolce_plus_genome_dataset(args)
    else:
        raise NotImplementedError
