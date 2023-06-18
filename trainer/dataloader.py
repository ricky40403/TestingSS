import math
import torch
from utils.dist import get_world_size
from trainer.sampler import IterationBasedBatchSampler
from trainer import transforms as T


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image
    """
    def __init__(self, tensors, image_sizes, pad_sizes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
        """
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.pad_sizes = pad_sizes

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes, self.pad_sizes)


def to_image_list(tensors, size_divisible=0, max_size=None):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        if tensors.dim() == 3:
            tensors = tensors[None]
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes, image_sizes)
    elif isinstance(tensors, (tuple, list)):
        if max_size is None:
            max_size = tuple(
                max(s) for s in zip(*[img.shape for img in tensors]))
        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors), ) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()  # + 114
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[:img.shape[0], :img.shape[1], :img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in tensors]
        pad_sizes = [batched_imgs.shape[-2:] for im in batched_imgs]

        return ImageList(batched_imgs, image_sizes, pad_sizes)
    else:
        raise TypeError('Unsupported type for to_image_list: {}'.format(
            type(tensors)))


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """
    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids

def make_batch_sampler(dataset,
                       sampler,
                       images_per_batch,
                       num_iters=None,
                       start_iter=0,
                       mosaic_warpper=False):
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler,
                                                          images_per_batch,
                                                          drop_last=False)
    if num_iters is not None:
        batch_sampler = IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter, enable_mosaic=mosaic_warpper)
    return batch_sampler

def build_transforms(start_epoch,
                     total_epochs,
                     no_aug_epochs,
                     iters_per_epoch,
                     num_workers,
                     batch_size,
                     num_gpus,
                     image_max_range=(640, 640),
                     flip_prob=0.5,
                     image_mean=[0, 0, 0],
                     image_std=[1., 1., 1.],
                     autoaug_dict=None,
                     keep_ratio=True):

    transform = [
        T.Resize(image_max_range, keep_ratio=keep_ratio),
        T.RandomHorizontalFlip(flip_prob),
        T.ToTensor(),
        T.Normalize(mean=image_mean, std=image_std),
    ]

    transform = T.Compose(transform)

    return transform

def build_dataloader(datasets,
                     augment,
                     batch_size=128,
                     start_epoch=None,
                     total_epochs=None,
                     no_aug_epochs=0,
                     is_train=True,
                     num_workers=8,
                     size_div=32):

    num_gpus = get_world_size()
    assert (
            batch_size % num_gpus == 0
        ), 'training_imgs_per_batch ({}) must be divisible by the number ' \
        'of GPUs ({}) used.'.format(batch_size, num_gpus)
    images_per_gpu = batch_size // num_gpus

    if is_train:
        iters_per_epoch = math.ceil(len(datasets) / batch_size)
        shuffle = True
        num_iters = total_epochs * iters_per_epoch
        start_iter = start_epoch * iters_per_epoch
    else:
        iters_per_epoch = math.ceil(len(datasets) / batch_size)
        shuffle = False
        num_iters = None
        start_iter = 0

    # transforms = augment.transform
    # enable_mosaic_mixup = 'mosaic_mixup' in augment
    transforms = None
    enable_mosaic_mixup = False

    transforms = build_transforms(start_epoch, total_epochs, no_aug_epochs,
                                  iters_per_epoch, num_workers, batch_size,
                                  num_gpus)

    # for dataset in datasets:
    #     dataset._transforms = transforms
    #     if hasattr(dataset, '_dataset'):
    #         dataset._dataset._transforms = transforms

    sampler = torch.utils.data.RandomSampler(datasets)
    batch_sampler = make_batch_sampler(datasets, sampler, images_per_gpu,
                                           num_iters, start_iter,
                                           enable_mosaic_mixup)

    collator = BatchCollator(size_div)
    data_loader = torch.utils.data.DataLoader(
        datasets,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=collator,
    )
    return data_loader

    # data_loaders = []
    # for dataset in datasets:
    #     sampler = make_data_sampler(dataset, shuffle)
    #     batch_sampler = make_batch_sampler(dataset, sampler, images_per_gpu,
    #                                        num_iters, start_iter,
    #                                        enable_mosaic_mixup)
    #     collator = BatchCollator(size_div)
    #     data_loader = torch.utils.data.DataLoader(
    #         dataset,
    #         num_workers=num_workers,
    #         batch_sampler=batch_sampler,
    #         collate_fn=collator,
    #     )
    #     data_loaders.append(data_loader)
    # if is_train:
    #     assert len(
    #         data_loaders) == 1, 'multi-training set is not supported yet!'
    #     return data_loaders[0]
    # return data_loaders