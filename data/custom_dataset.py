"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from data.pix2pix_dataset import Pix2pixDataset
from data.image_folder import make_dataset


class CustomDataset(Pix2pixDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir, --instance_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        if is_train:
            load_size = 284 if is_train else 256
            parser.set_defaults(load_size=load_size)
            parser.set_defaults(crop_size=256)
            parser.set_defaults(display_winsize=256)
        else:
            load_size = 256
            parser.set_defaults(load_size=load_size)
            parser.set_defaults(crop_size=load_size)
            parser.set_defaults(display_winsize=load_size)

        parser.add_argument('--origin_dir', type=str, required=True)
        parser.add_argument('--image_dir', type=str, default='')
        parser.add_argument('--masks_dir', type=str, default='')
        parser.add_argument('--mask_dir', type=str, default='')
        return parser

    def get_paths(self, opt):
        origin_dir = opt.origin_dir
        origin_paths = make_dataset(origin_dir, recursive=False, read_cache=True)

        if len(opt.image_dir) > 0:
            image_dir = opt.image_dir
            image_paths = make_dataset(image_dir, recursive=False, read_cache=True)
        else:
            image_paths = []

        if len(opt.masks_dir) > 0:
            masks_dir = opt.masks_dir
            masks_paths = make_dataset(masks_dir, recursive=False, read_cache=True)
        else:
            masks_paths = []

        if opt.isTrain:
            assert len(origin_paths) == len(image_paths), "The #images in %s and %s do not match. Is there something wrong?"
            mask_dir = opt.mask_dir
            mask_paths = make_dataset(mask_dir, recursive=False, read_cache=True)

            return origin_paths, image_paths, masks_paths, mask_paths

        else:
            return origin_paths, image_paths, masks_paths
