from dataloading.square_patch_dataset import load_square_patch_dataset
from dataloading.full_image_dataset import load_full_image_dataset

def get_dataloaders(args):

    if args.data == "square_patch":
        return load_square_patch_dataset(args)
    if args.data == "full_image":
        return load_full_image_dataset(args)
    #if args.data == "slice_dolce":
    #    return load_slice_dolce_dataset(args)
    else:
        raise NotImplementedError
