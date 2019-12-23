import os
import math
import time
import torch
import random
import datetime
import itertools
import scipy.misc as m
import torch.nn as nn
from tensorboardX import SummaryWriter
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from utils import *

from SUPER_RESOLUTION.config import *
from SUPER_RESOLUTION.FHN_loader import CelebA_HQ_loader
from SUPER_RESOLUTION.model.utils import init_log, AverageMeter
from SUPER_RESOLUTION.loss.loss import Landmark_Loss, CrossEntropyLoss2d

from SUPER_RESOLUTION.model.FSRnet import Coarse_SR_Network, Fine_SR_Decoder, Fine_SR_Encoder, Prior_Estimation_Network


def main():
    torch.manual_seed(SEED)

    if not os.path.exists(CHECKPOINT):
        os.mkdir(CHECKPOINT)
    if not os.path.exists(RESULT):
        os.mkdir(RESULT)

    logging = init_log(CHECKPOINT)

    # ************************      Dataloader      ***********************
    trainset = CelebA_HQ_loader(parsing_root=PARSING_ROOT, landmark_root=LANDMARK_ROOT, img_root=IMG_ROOT,
                                is_transform=True)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                                  drop_last=True)

    # *************************     Model       **************************
    coarse = Coarse_SR_Network()
    prior_estimation = Prior_Estimation_Network()
    encoder = Fine_SR_Encoder()
    decoder = Fine_SR_Decoder()

    # ***************        Optimizer       *****************
    optim_coarse = torch.optim.Adam(coarse.parameters(), lr=LR, weight_decay=WD, betas=(BETA1, BETA2))
    # optim_encoder = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WD, betas=(BETA1, BETA2))
    # optim_decoder = torch.optim.Adam(decoder.parameters(), lr=LR, weight_decay=WD, betas=(BETA1, BETA2))
    optim_prior_estimation = torch.optim.Adam(prior_estimation.parameters(), lr=LR, weight_decay=WD,
                                              betas=(BETA1, BETA2))
    optim_encoder_decoder = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=LR,
                                             weight_decay=WD, betas=(BETA1, BETA2))

    # ****************      LR Scheduler        *******************
    sche_coarse = torch.optim.lr_scheduler.MultiStepLR(optim_coarse, MILESTONES, gamma=0.1)
    # sche_encoder = torch.optim.lr_scheduler.MultiStepLR(optim_encoder, MILESTONES, gamma=0.1)
    # sche_decoder = torch.optim.lr_scheduler.MultiStepLR(optim_decoder, MILESTONES, gamma=0.1)
    sche_prior_estimation = torch.optim.lr_scheduler.MultiStepLR(optim_prior_estimation, MILESTONES, gamma=0.1)
    sche_encoder_decoder = torch.optim.lr_scheduler.MultiStepLR(optim_encoder_decoder, MILESTONES, gamma=0.1)

    # ***************************       Resume      *********************
    if not RESUME_COARSE:
        coarse.apply(weights_init)
        prior_estimation.apply(weights_init)
        encoder.apply(weights_init)
        decoder.apply(weights_init)
        start_epoch = 0
    else:
        # load model
        checkpoint = torch.load(RESUME_COARSE)
        coarse.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        checkpoint = torch.load(RESUME_ENCODER)
        encoder.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(RESUME_DECODER)
        decoder.load_state_dict(checkpoint['state_dict'])

        checkpoint = torch.load(RESUME_PRIOR_ESTIMATION)
        prior_estimation.load_state_dict(checkpoint['state_dict'])

        # load optimizer
        checkpoint = torch.load(RESUME_OPTIM_COARSE)
        optim_coarse.load_state_dict(checkpoint)
        for state in optim_coarse.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        # checkpoint = torch.load(RESUME_OPTIM_ENCODER)
        # optim_encoder.load_state_dict(checkpoint)
        # for state in optim_encoder.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()
        #
        # checkpoint = torch.load(RESUME_OPTIM_DECODER)
        # optim_decoder.load_state_dict(checkpoint)
        # for state in optim_decoder.state.values():
        #     for k, v in state.items():
        #         if isinstance(v, torch.Tensor):
        #             state[k] = v.cuda()

        checkpoint = torch.load(RESUME_OPTIM_PRIOR_ESTIMATION)
        optim_prior_estimation.load_state_dict(checkpoint)
        for state in optim_prior_estimation.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        checkpoint = torch.load(RESUME_OPTIM_ENCODER_DECODER)
        optim_encoder_decoder.load_state_dict(checkpoint)
        for state in optim_encoder_decoder.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    # ***********************       Loss        *********************
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    landmark_loss = Landmark_Loss()
    cross_entropy_loss = CrossEntropyLoss2d()

    # **********************      cuda        **********************
    coarse.to(DEVICE)
    prior_estimation.to(DEVICE)
    encoder.to(DEVICE)
    decoder.to(DEVICE)

    l1_loss.to(DEVICE)
    mse_loss.to(DEVICE)
    landmark_loss.to(DEVICE)
    cross_entropy_loss.to(DEVICE)

    # ***********************       Summary Writer      ***********************
    writer_coarse = SummaryWriter(os.path.join(LOGS, 'coarse'))
    writer_prior_estimation = SummaryWriter(os.path.join(LOGS, 'prior_estimation'))
    writer_encoder = SummaryWriter(os.path.join(LOGS, 'encoder'))
    writer_decoder = SummaryWriter(os.path.join(LOGS, 'decoder'))

    count = int(len(trainloader) // SAVE_IMG)

    for epoch in range(start_epoch, EPOCHS):
        losses_coarse = AverageMeter()
        losses_prior_estimation = AverageMeter()
        losses_encoder = AverageMeter()
        losses_decoder = AverageMeter()
        losses_encoder_decoder = AverageMeter()
        batch_time = AverageMeter()

        sche_coarse.step(epoch)
        # sche_encoder.step(epoch)
        # sche_decoder.step(epoch)
        sche_prior_estimation.step(epoch)
        sche_encoder_decoder.step(epoch)

        bar = Bar('Processing: ', max=len(trainloader))

        prev_time = time.time()
        end_time = time.time()

        for i, batch in enumerate(tqdm(trainloader)):
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            lr_img = batch['lr_img']
            hr_img = batch['hr_img']
            parsing_label = batch['parsing_label']
            heatmap = batch['heatmap']

            # ********      coarse      *******
            coarse_img, coarse_feature = coarse(lr_img)
            coarse_loss = mse_loss(coarse_img, hr_img)

            optim_coarse.zero_grad()
            coarse_loss.backward()
            optim_coarse.step()
            losses_coarse.update(coarse_loss.data.cpu().numpy(), BATCH_SIZE)

            # *******       prior estimation        *******
            prior_feature, parsing_out, landmark_out = prior_estimation(coarse_img)
            prior_loss = landmark_loss(landmark_out, heatmap) + cross_entropy_loss(parsing_out, parsing_label)

            optim_prior_estimation.zero_grad()
            prior_loss.backward()
            optim_prior_estimation.step()
            losses_prior_estimation.update(prior_loss.data.cpu().numpy(), BATCH_SIZE)

            # *******       encoder & decoder     *******
            encoder_feature = encoder(coarse_feature)
            feature = torch.cat((prior_feature, encoder_feature))
            sr_img = decoder(feature)

            encoder_decoder_loss = mse_loss(sr_img, hr_img)
            optim_encoder_decoder.zero_grad()
            encoder_decoder_loss.backward()
            optim_encoder_decoder.step()

            # Determine approximate time left.
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            batchs_done = epoch * len(trainloader) + i
            batchs_left = EPOCHS * len(trainloader) - batchs_done
            time_left = datetime.timedelta(seconds=batchs_left * (time.time() - prev_time))
            prev_time = time.time()

            bar.suffix = 'Epoch/Step: {epoch}/{step} | Coarse: {coarse:.4f} | Prior Estimation: {prior:.4f} | ' \
                         'Encoder: {encoder:.4f} | Decoder: {decoder:.4f} | ETA: {time_left}'.format(
                epoch=epoch,
                step=i,
                coarse=losses_coarse.avg,
                prior=losses_prior_estimation.avg,
                encoder=losses_encoder_decoder.avg,
                decoder=losses_encoder_decoder.avg,
                time_left=time_left
            )
            print(bar.suffix)

            # save image
            if i % count == 0:
                img = lr_img.detach().cpu().numpy()[0]
                img_name = 'lr_img_{}_{}'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))

                img = hr_img.detach().cpu().numpy()[0]
                img_name = 'hr_img_{}_{}'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))

                img = sr_img.detach().cpu().numpy()[0]
                img_name = 'sr_img_{}_{}'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))

                landmark_out = landmark_out.detach().cpu().numpy()[0]
                img = np.sum(landmark_out, axis=0)
                img_name = 'landmark_{}_{}'.format(epoch, i // count)
                plt.imshow(img)
                plt.savefig(os.path.join(RESULT, img_name))

                img = parsing_out.detach().cpu().numpy()[0].max(dim=1)[1]
                img = trainset.decode_segmap(img)
                img_name = 'parsing_{}_{}'.format(epoch, i // count)
                plt.imsave(os.path.join(RESULT, img_name),img)

def weights_init(m):
    for each in m.modules():
        if isinstance(each, nn.Conv2d):
            nn.init.xavier_uniform_(each.weight.data)
            if each.bias is not None:
                each.bias.data.zero_()
        elif isinstance(each, nn.BatchNorm2d):
            each.weight.data.fill_(1)
            each.bias.data.zero_()
        elif isinstance(each, nn.Linear):
            nn.init.xavier_uniform_(each.weight.data)
            each.bias.data.zero_()

if __name__ == "__main__":
    main()
