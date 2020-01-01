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
import torchvision
import cv2

from utils import *

from SUPER_RESOLUTION.config import *
from SUPER_RESOLUTION.FHN_loader import CelebA_HQ_loader
from SUPER_RESOLUTION.model.utils import init_log, AverageMeter
from SUPER_RESOLUTION.loss.loss import Landmark_Loss, CrossEntropyLoss2d
from SUPER_RESOLUTION.model.model_irse import IR_50

from SUPER_RESOLUTION.model.FSRnet import Coarse_SR_Network, Fine_SR_Decoder, Fine_SR_Encoder, Prior_Estimation_Network
from SUPER_RESOLUTION.model.GroupDepthConv import FeatureExtractor

from SUPER_RESOLUTION.align.detector import detect_faces
from SUPER_RESOLUTION.align.align_trans import get_reference_facial_points, warp_and_crop_face


def get_feature(hr_imgs, imgs, backbone, count):
    tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    original_5_points = get_reference_facial_points(default_square=True)
    reference = (original_5_points) * 1.0
    crop_size = 112
    hr_imgs = hr_imgs.detach()
    hr_warped_faces = torch.ones((BATCH_SIZE, 3, 112, 112))
    hr_warped_faces = hr_warped_faces.to(DEVICE)
    
    imgs = imgs.detach()
    warped_faces = torch.ones((BATCH_SIZE, 3, 112, 112))
    warped_faces = warped_faces.to(DEVICE)
    for i in range(BATCH_SIZE):
        hr_img = hr_imgs[i]
        # img = Image.fromarray(img)
        # img = torchvision.transforms.ToPILImage()(img).convert('RGB')
        hr_img = hr_img.transpose(0, 1).transpose(1, 2)
        hr_img = (hr_img * 0.5 + 0.5) * 255
        hr_img = hr_img.cpu().numpy()
        hr_img = Image.fromarray(hr_img.astype(np.uint8))
        
        img = imgs[i]
        # img = Image.fromarray(img)
        # img = torchvision.transforms.ToPILImage()(img).convert('RGB')
        img = img.transpose(0, 1).transpose(1, 2)
        img = (img * 0.5 + 0.5) * 255
        img = img.cpu().numpy()
        img = Image.fromarray(img.astype(np.uint8))
        
        try:
            _, landmarks = detect_faces(hr_img)
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            hr_warped_face = warp_and_crop_face(np.array(hr_img), facial5points, reference, crop_size=(crop_size, crop_size))
            warped_face = warp_and_crop_face(np.array(img), facial5points, reference, crop_size=(crop_size, crop_size))
            # warped_face = torch.from_numpy(warped_face[0])
            hr_warped_face = tf(hr_warped_face[0])
            warped_face = tf(warped_face[0])
            hr_warped_faces[i] = hr_warped_face
            warped_faces[i] = warped_face
        except:
            hr_warped_faces[i] = tf(hr_img)
            warped_faces[i] = tf(img)
            count += 1
    feature = backbone(warped_faces)
    hr_feature = backbone(hr_warped_faces)
    return hr_feature, feature, count


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
    backbone = IR_50([112, 112])
    backbone.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    
    fe = FeatureExtractor()
    
    layer_list = list(backbone.body._modules.keys())[-3:-1]
    
    # ***************        Optimizer       *****************
    optim_coarse = torch.optim.Adam(coarse.parameters(), lr=LR_COARSE, weight_decay=WD, betas=(BETA1, BETA2))
    # optim_encoder = torch.optim.Adam(encoder.parameters(), lr=LR, weight_decay=WD, betas=(BETA1, BETA2))
    # optim_decoder = torch.optim.Adam(decoder.parameters(), lr=LR, weight_decay=WD, betas=(BETA1, BETA2))
    optim_prior_estimation = torch.optim.Adam(prior_estimation.parameters(), lr=LR_PRIOR, weight_decay=WD,
                                              betas=(BETA1, BETA2))
    optim_encoder_decoder = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=LR_ENCODER_DECODER,
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
    backbone.to(DEVICE)
    
    l1_loss.to(DEVICE)
    mse_loss.to(DEVICE)
    landmark_loss.to(DEVICE)
    cross_entropy_loss.to(DEVICE)
    
    # ***********************       Summary Writer      ***********************
    writer_coarse = SummaryWriter(os.path.join(LOGS, 'coarse'))
    writer_prior_estimation = SummaryWriter(os.path.join(LOGS, 'prior_estimation'))
    # writer_encoder = SummaryWriter(os.path.join(LOGS, 'encoder'))
    # writer_decoder = SummaryWriter(os.path.join(LOGS, 'decoder'))
    writer_encoder_decoder = SummaryWriter(os.path.join(LOGS, 'encoder_decoder'))
    
    count = int(len(trainloader) // SAVE_IMG)
    
    for epoch in range(start_epoch, EPOCHS):
        losses_coarse = AverageMeter()
        losses_prior_estimation = AverageMeter()
        # losses_encoder = AverageMeter()
        # losses_decoder = AverageMeter()
        losses_encoder_decoder = AverageMeter()
        batch_time = AverageMeter()
        
        coarse.train()
        encoder.train()
        decoder.train()
        prior_estimation.train()
        backbone.eval()
        
        sche_coarse.step(epoch)
        # sche_encoder.step(epoch)
        # sche_decoder.step(epoch)
        sche_prior_estimation.step(epoch)
        sche_encoder_decoder.step(epoch)
        
        bar = Bar('Processing: ', max=len(trainloader))
        
        prev_time = time.time()
        end_time = time.time()
        
        num = 0
        
        for i, batch in enumerate(tqdm(trainloader)):
            for k in batch:
                batch[k] = batch[k].to(DEVICE)
            lr_img = batch['lr_img']
            hr_img = batch['hr_img']
            parsing_label = batch['parsing_label']
            heatmap = batch['heatmap']
            
            # ********      coarse      *******
            coarse_img = coarse(lr_img)
            # hr_feature, feature, num = get_feature(hr_img, coarse_img, backbone, num)
            # coarse_loss = mse_loss(feature, hr_feature) * LAMBDA_FEATURE + mse_loss(coarse_img, hr_img) * LAMBDA_PIXEL
            
            features = backbone.input_layer(hr_img)
            logits_dict, _, _, _ = fe(features, layer_list, backbone.body)
            
            features = backbone.input_layer(coarse_img)
            coarse_logits_dict, _, _, _ = fe(features, layer_list, backbone.body)
            
            recon_loss = 0.
            
            for layer in layer_list:
                temp_loss = mse_loss(logits_dict[layer], coarse_logits_dict[layer])
                recon_loss += temp_loss * LAMBDA_FEATURE
            
            optim_coarse.zero_grad()
            recon_loss.backward(retain_graph=True)
            optim_coarse.step()
            losses_coarse.update(recon_loss.data.cpu().numpy(), BATCH_SIZE)
            
            if epoch > WARMING_UP:
                # *******       prior estimation        *******
                prior_feature, landmark_out, parsing_out = prior_estimation(coarse_img)
                
                # *******       encoder & decoder     *******
                encoder_feature = encoder(coarse_img)
                feature = torch.cat((prior_feature, encoder_feature), dim=1)
                sr_img = decoder(feature)
                
                prior_loss = landmark_loss(landmark_out, heatmap) * LAMBDA_LANDMARK + cross_entropy_loss(parsing_out, parsing_label) * LAMBDA_PARSING
                
                optim_prior_estimation.zero_grad()
                prior_loss.backward(retain_graph=True)
                optim_prior_estimation.step()
                losses_prior_estimation.update(prior_loss.data.cpu().numpy(), BATCH_SIZE)
                
                # hr_feature, feature, _ = get_feature(hr_img, sr_img, backbone, num)
                # sr_feature = backbone(sr_img)
                # hr_feature = backbone(hr_img)
                # encoder_decoder_loss = mse_loss(feature, hr_feature) * LAMBDA_FEATURE + mse_loss(sr_img, hr_img) * LAMBDA_PIXEL

                features = backbone.input_layer(hr_img)
                logits_dict, _, _, _ = fe(features, layer_list, backbone.body)

                features = backbone.input_layer(sr_img)
                sr_logits_dict, _, _, _ = fe(features, layer_list, backbone.body)

                recon_loss = 0.

                for layer in layer_list:
                    temp_loss = mse_loss(logits_dict[layer], sr_logits_dict[layer])
                    recon_loss += temp_loss * LAMBDA_FEATURE
                    
                optim_encoder_decoder.zero_grad()
                recon_loss.backward()
                optim_encoder_decoder.step()
                losses_encoder_decoder.update(recon_loss.data.cpu().numpy(), BATCH_SIZE)
            
            # Determine approximate time left.
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            batchs_done = epoch * len(trainloader) + i
            batchs_left = EPOCHS * len(trainloader) - batchs_done
            time_left = datetime.timedelta(seconds=batchs_left * (time.time() - prev_time))
            prev_time = time.time()
            
            bar.suffix = '  Epoch/Step: {epoch}/{step} | Coarse: {coarse:.4f} | Prior Estimation: {prior:.4f} | ' \
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
                img_name = 'lr_img_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                img = coarse_img.detach().cpu().numpy()[0]
                img_name = 'coarse_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                img = hr_img.detach().cpu().numpy()[0]
                img_name = 'hr_img_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                if epoch > WARMING_UP:
                    img = sr_img.detach().cpu().numpy()[0]
                    img_name = 'sr_img_{}_{}.jpg'.format(epoch, i // count)
                    image = m.toimage(img, cmin=None, cmax=None)
                    image.save(os.path.join(RESULT, img_name))
                    
                    landmark_out = landmark_out.detach().cpu().numpy()[0]
                    # img = landmark_out
                    img = np.sum(landmark_out, axis=0)
                    img_name = 'landmark_{}_{}.jpg'.format(epoch, i // count)
                    plt.axis('off')
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.imshow(img)
                    plt.savefig(os.path.join(RESULT, img_name), transparent=True, pad_inches=0)
                    
                    img = parsing_out[0].max(0)[1].detach().cpu().numpy()
                    # img = parsing_label[0].detach().cpu().numpy()
                    # img = np.expand_dims(img, axis=0)
                    # unloader = torchvision.transforms.ToPILImage()
                    # img = unloader(img)
                    img = trainset.decode_segmap(img)
                    img_name = 'parsing_{}_{}.jpg'.format(epoch, i // count)
                    
                    plt.imsave(os.path.join(RESULT, img_name), img)
        
        writer_coarse.add_scalar('coarse', losses_coarse.avg, epoch)
        writer_prior_estimation.add_scalar('prior_estimation', losses_prior_estimation.avg, epoch)
        writer_encoder_decoder.add_scalar('encoder_decoder', losses_encoder_decoder.avg, epoch)
        
        torch.save({'state_dict': coarse.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'coarse_' + FLAG + '.pth'))
        torch.save({'state_dict': prior_estimation.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'prior_estimation_' + FLAG + '.pth'))
        torch.save({'state_dict': encoder.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'encoder_' + FLAG + '.pth'))
        torch.save({'state_dict': decoder.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'decoder_' + FLAG + '.pth'))
        
        torch.save(optim_coarse.state_dict(), os.path.join(CHECKPOINT, 'optim_coarse_' + FLAG + '.pth'))
        torch.save(optim_encoder_decoder.state_dict(), os.path.join(CHECKPOINT, 'optim_encoder_decoder_' + FLAG + '.pth'))
        torch.save(optim_prior_estimation.state_dict(), os.path.join(CHECKPOINT, 'optim_prior_estimation_' + FLAG + '.pth'))


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
