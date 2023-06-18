import time
import yaml
import argparse

import torch

from models import build_models
from data.coco import COCODataset
from trainer.dataloader import build_dataloader, to_image_list
from trainer.scheduler import cosine_scheduler
from trainer.optimizer import get_optimizer
from trainer import transforms as T

def make_parser():
    """
    Create a parser with some common arguments used by users.

    Returns:
        argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser('Damo-Yolo train parser')

    parser.add_argument(
        '-c',
        '--config_file',
        default=None,
        type=str,
        help='plz input your config file',
    )
    return parser



def main():
    args = make_parser().parse_args()
    with open(args.config_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    

    batch_num = 8
    total_epoch = 10
    warmup_epochs= 1
    start_iter = 0
    device = "cuda"
    model = build_models(cfg)
    model.to(device)
    image_max_range=(640, 640)
    keep_ratio=False
    transform = [
        T.Resize(image_max_range, keep_ratio=keep_ratio),
        T.ToTensor(),
        T.Normalize(mean=[0, 0, 0], std=[1., 1., 1.]),
    ]

    transform = T.Compose(transform)
    train_dataset = COCODataset("datasets/coco/annotations/instances_train2017.json", "datasets/coco/train2017", transforms = transform)
    train_loader = build_dataloader(train_dataset, None, start_epoch = 0, total_epochs = total_epoch, batch_size=batch_num)
    total_iters = len(train_dataset) * total_epoch
    optimizer = get_optimizer(cfg, model)
    lr_scheduler = cosine_scheduler(0.001 / 64, batch_num, 0.05, total_iters, 0, 100, 0)

    model.train()    
    iter_start_time = time.time()
    iter_end_time = time.time()
    for data_iter, (inps, targets, ids) in enumerate(train_loader):
        cur_iter = start_iter + data_iter

        lr = lr_scheduler.get_lr(cur_iter)
        print("lr" , lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        inps = inps.to(device)  # ImageList: tensors, img_size
        targets = [target.to(device)
                    for target in targets]  # BoxList: bbox, num_boxes ...

        model_start_time = time.time()
        images = to_image_list(inps)
        outputs = model(images.tensors, targets)
        loss = outputs['total_loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        iter_start_time = iter_end_time
        iter_end_time = time.time()
        print(outputs)


    # x = torch.randn(1, 3, 416, 416, requires_grad=True)
    # torch.onnx.export(
    #     model,
    #     x,
    #     "test.onnx",
    #     export_params=True,
    #     opset_version=10,
    #     # do_constant_folding=True,
    # )



if __name__ == '__main__':
    main()
