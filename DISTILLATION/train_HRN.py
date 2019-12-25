import torch
import torch.utils.data as data

import os
import numpy as np
import time
from utils import *
from tqdm import tqdm
from tensorboardX import SummaryWriter

from DISTILLATION.MS1M_loader import MS1M_loader
from DISTILLATION.config import *
from DISTILLATION.model.utils import init_log, AverageMeter
from DISTILLATION.model.model_irse import IR_50

from utils.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, \
    buffer_val, AverageMeter, accuracy


def main():
    torch.manual_seed(SEED)

    if not os.path.exists(CHECKPOINT):
        os.mkdir(CHECKPOINT)

    logging = init_log(CHECKPOINT)
    _print = logging.critical

    # *****************     Dataloader      ****************
    train_transform = transforms.Compose([
        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * INPUT_SIZE[0] / 112), int(128 * INPUT_SIZE[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([INPUT_SIZE[0], INPUT_SIZE[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
    ])

    # dataset_train = datasets.ImageFolder(os.path.join(DATA_ROOT, 'ms1m_align_112', 'imgs'), train_transform)
    dataset_train = MS1M_loader(os.path.join(DATA_ROOT, 'ms1m_align_112', 'imgs'), train_transform)
    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, sampler=sampler, pin_memory=PIN_MEMORY,
        num_workers=NUM_WORKERS, drop_last=DROP_LAST
    )

    NUM_CLASS = len(train_loader.dataset.classes)
    _print("Number of Training Classes: {}".format(NUM_CLASS))

    lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(
        DATA_ROOT)

    # ****************      Model       ****************
    teacher = IR_50(INPUT_SIZE)
    student = IR_50(INPUT_SIZE)
    assistant = IR_50(INPUT_SIZE)

    teacher.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    student.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    assistant.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))

    teacher.to(DEVICE)
    student.to(DEVICE)
    assistant.to(DEVICE)

    # ****************      Optimizer       ****************
    student_params_only_bn, student_params_wo_bn = separate_irse_bn_paras(student)
    assistant_params_only_bn, assistant_params_wo_bn = separate_irse_bn_paras(assistant)

    student_optimizer = torch.optim.SGD(
        [{'params': student_params_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': student_params_only_bn}], lr=LR,
        momentum=MOMENTUM)
    assistant_optimizer = torch.optim.SGD(
        [{'params': assistant_params_wo_bn, 'weight_decay': WEIGHT_DECAY}, {'params': assistant_params_only_bn}], lr=LR,
        momentum=MOMENTUM)

    # ****************      Scheduler       ****************
    student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, milestones=MILESTONES, gamma=0.1)
    assistant_scheduler = torch.optim.lr_scheduler.MultiStepLR(assistant_optimizer, milestones=MILESTONES, gamma=0.1)

    # ****************      Resume      *****************
    if RESUME_ROOT:
        student.load_state_dict(torch.load(os.path.join(RESUME_ROOT, 'student_IR_50.pth'))['state_dict'])
        assistant.load_state_dict(torch.load(os.path.join(RESUME_ROOT, 'assistant_IR_50.pth'))['state_dict'])
        start_epoch = torch.load(os.path.join(RESUME_ROOT, 'assistant_IR_50.pth'))['epoch']
    else:
        start_epoch = 0

    # *****************     Warm Up     ****************
    NUM_EPOCH_WARM_UP = EPOCHS // 25
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP
    batch_index = 0

    for epoch in range(start_epoch, EPOCHS):
        student.train()
        assistant.train()
        teacher.eval()

        losses = AverageMeter()

        for batch in tqdm(iter(train_loader)):
            sr_img = batch['sr_img']
            hr_img = batch['hr_img']
            sr_img.to(DEVICE)
            hr_img.to(DEVICE)

            if (epoch + 1 <= NUM_EPOCH_WARM_UP) and (batch + 1) <= NUM_BATCH_WARM_UP:
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, student_optimizer)
                warm_up_lr(batch + 1, NUM_BATCH_WARM_UP, LR, assistant_optimizer)

            teacher_feature = teacher(hr_img)
            student_feature = student(sr_img)


        student_scheduler.step(epoch)
        assistant_scheduler.step(epoch)


if __name__ == '__main__':
    main()
