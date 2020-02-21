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
import shutil
import cv2
from collections import OrderedDict
from torch.autograd import Variable

from utils import *

from SUPER_RESOLUTION.config_GAN import *
from SUPER_RESOLUTION.FHN_loader import Scface_loader, Scface_loader_V2, LFW_loader, FaceSurf_loader
from SUPER_RESOLUTION.model.utils import init_log, AverageMeter
from SUPER_RESOLUTION.loss.loss import Landmark_Loss, CrossEntropyLoss2d
from SUPER_RESOLUTION.model.model_irse import IR_50
from SUPER_RESOLUTION.model.lightcnn import LightCNN_29Layers_v2

from SUPER_RESOLUTION.model.FSRnet import Coarse_SR_Network, Fine_SR_Decoder, Fine_SR_Encoder, Prior_Estimation_Network, LR_Generator, Discriminator, Discriminator_CLS
from SUPER_RESOLUTION.model.GroupDepthConv import FeatureExtractor_lightcnn, FeatureExtractor

from SUPER_RESOLUTION.align.detector import detect_faces
from SUPER_RESOLUTION.align.align_trans import get_reference_facial_points, warp_and_crop_face


def get_feature(imgs, backbone, backbone_name, layer_list):
    # tf = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.Resize([128, 128]),
    #         torchvision.transforms.Grayscale(num_output_channels=1),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize([0.5], [0.5]),
    #     ]
    # )
    # feature = backbone(tf(imgs))
    if backbone_name == 'lightcnn':
        fe = FeatureExtractor_lightcnn()
        pooling_list = ['conv1', 'group1', 'group2', 'group4']
        logits_dict, _, _, _ = fe(F.interpolate(rgb2gray(imgs), 128), layer_list, pooling_list, backbone)
    elif backbone_name == 'IR50':
        fe = FeatureExtractor()
        features = backbone.input_layer(imgs)
        logits_dict, _, _, _ = fe(features, layer_list, backbone.body)
    return logits_dict


def rgb2gray(imgs):
    gray_imgs = torch.zeros((imgs.size(0), 1, imgs.size(2), imgs.size(3))).cuda()
    for i in range(BATCH_SIZE):
        temp = imgs[i]
        R = temp[0]
        G = temp[1]
        B = temp[2]
        tensor = 0.299 * R + 0.587 * G + 0.114 * B
        tensor = tensor.unsqueeze(0)
        gray_imgs[i] = tensor
    return gray_imgs


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


def label2onehot(labels, dim, batch_size=BATCH_SIZE):
    out = torch.zeros(batch_size, dim)
    out[np.arange(batch_size), labels] = 1
    return out


def main():
    torch.manual_seed(SEED)
    
    if not os.path.exists(CHECKPOINT):
        os.mkdir(CHECKPOINT)
    if not os.path.exists(RESULT):
        os.mkdir(RESULT)
    if os.path.exists(LOGS):
        shutil.rmtree(LOGS)
    
    tf = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize([128, 128]),
            torchvision.transforms.Grayscale(num_output_channels=1),
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    # ************************      Dataloader      ***********************
    # trainset = Scface_loader_V2(img_root=IMG_ROOT, sr_img_root=SR_IMG_ROOT, is_transform=True)
    # trainset = LFW_loader(img_root=IMG_ROOT)
    trainset = FaceSurf_loader(lr_img_root=IMG_ROOT, hr_img_root=SR_IMG_ROOT, openset_path=OPENSET_PATH, is_transform=True)
    trainloader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    
    # *************************     Model       **************************
    coarse = Coarse_SR_Network(input_dim=3)
    prior_estimation = Prior_Estimation_Network()
    encoder = Fine_SR_Encoder()
    decoder = Fine_SR_Decoder()
    g_lr = LR_Generator()
    d_lr = Discriminator_CLS()
    d_sr = Discriminator()
    
    if BACKBONE_NAME == 'lightcnn':
        backbone = LightCNN_29Layers_v2(num_classes=80013)
        # checkpoint = torch.load(BACKBONE_RESUME_ROOT)['state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in checkpoint.items():
        #     name = k[7:]  # remove 'module'
        #     new_state_dict[name] = v
        # backbone.load_state_dict(new_state_dict)
        
        layer_list = list(backbone._modules.keys())[:-2]
        
        pooling_list = ['conv1', 'group1', 'group2', 'group4']
        
        input = Variable(torch.rand(2, 1, 128, 128), requires_grad=False)
        fe = FeatureExtractor_lightcnn()
        
        out, _, _, _ = fe(input, layer_list, pooling_list, backbone)
        feature_dict = {}
        for k, v in out.items():
            feature_dict[k] = v.size()
        layer_list = list(feature_dict.keys())
        extract_layer_list = layer_list[-2:]
        extract_layer_list += layer_list[:2]
        print(layer_list)
    elif BACKBONE_NAME == 'IR50':
        backbone = IR_50([112, 112])
        backbone.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        
        fe = FeatureExtractor()
        layer_list = list(backbone.body._modules.keys())[-8:-1]
        # layer_list += list(backbone.body._modules.keys())[0:2]
        extract_layer_list = layer_list
    
    # ***************        Optimizer       *****************
    optim_coarse = torch.optim.Adam(itertools.chain(coarse.parameters(), g_lr.parameters()), lr=LR_COARSE, weight_decay=WD, betas=(BETA1, BETA2))
    # optim_prior_estimation = torch.optim.Adam(prior_estimation.parameters(), lr=LR_PRIOR, weight_decay=WD,
    #                                           betas=(BETA1, BETA2))
    # optim_encoder_decoder = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=LR_ENCODER_DECODER,
    #                                          weight_decay=WD, betas=(BETA1, BETA2))
    optim_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters(), g_lr.parameters()),
                               lr=LR_G, weight_decay=WD, betas=(BETA1, BETA2))
    optim_D_SR = torch.optim.Adam(d_sr.parameters(), lr=LR_D, betas=(D_BETA1, D_BETA2), weight_decay=WD)
    optim_D_LR = torch.optim.Adam(d_lr.parameters(), lr=LR_D, betas=(D_BETA1, D_BETA2), weight_decay=WD)
    
    # ****************      LR Scheduler        *******************
    sche_coarse = torch.optim.lr_scheduler.MultiStepLR(optim_coarse, MILESTONES, gamma=0.1)
    # sche_prior_estimation = torch.optim.lr_scheduler.MultiStepLR(optim_prior_estimation, MILESTONES, gamma=0.1)
    # sche_encoder_decoder = torch.optim.lr_scheduler.MultiStepLR(optim_encoder_decoder, MILESTONES, gamma=0.1)
    sche_G = torch.optim.lr_scheduler.MultiStepLR(optim_G, MILESTONES, gamma=0.1)
    sche_D_SR = torch.optim.lr_scheduler.MultiStepLR(optim_D_SR, MILESTONES, gamma=0.1)
    sche_D_LR = torch.optim.lr_scheduler.MultiStepLR(optim_D_LR, MILESTONES, gamma=0.1)
    
    if not RESUME_COARSE:
        coarse.apply(weights_init)
        encoder.apply(weights_init)
        decoder.apply(weights_init)
        d_lr.apply(weights_init)
        d_sr.apply(weights_init)
        g_lr.apply(weights_init)
        
        checkpoint = torch.load(RESUME_PRIOR_ESTIMATION)
        prior_estimation.load_state_dict(checkpoint['state_dict'])
        
        start_epoch = 0
    
    else:
        checkpoint = torch.load(RESUME_COARSE)
        coarse.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        checkpoint = torch.load(RESUME_ENCODER)
        encoder.load_state_dict(checkpoint['state_dict'])
        
        checkpoint = torch.load(RESUME_DECODER)
        decoder.load_state_dict(checkpoint['state_dict'])
        
        checkpoint = torch.load(RESUME_D_SR)
        decoder.load_state_dict(checkpoint['state_dict'])
        
        checkpoint = torch.load(RESUME_D_LR)
        decoder.load_state_dict(checkpoint['state_dict'])
        
        checkpoint = torch.load(RESUME_OPTIM_G)
        optim_G.load_state_dict(checkpoint)
        for state in optim_G.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        checkpoint = torch.load(RESUME_OPTIM_D_LR)
        optim_D_LR.load_state_dict(checkpoint)
        for state in optim_D_LR.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        
        checkpoint = torch.load(RESUME_OPTIM_D_SR)
        optim_D_SR.load_state_dict(checkpoint)
        for state in optim_D_SR.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
    
    # ***********************       Loss        *********************
    mse_loss = nn.MSELoss()
    l1_loss = nn.L1Loss()
    landmark_loss = Landmark_Loss()
    cross_entropy_loss = CrossEntropyLoss2d()
    bce_loss = torch.nn.BCELoss()
    
    # **********************      cuda        **********************
    coarse.to(DEVICE)
    prior_estimation.to(DEVICE)
    encoder.to(DEVICE)
    decoder.to(DEVICE)
    backbone.to(DEVICE)
    d_lr.to(DEVICE)
    d_sr.to(DEVICE)
    g_lr.to(DEVICE)
    
    l1_loss.to(DEVICE)
    mse_loss.to(DEVICE)
    landmark_loss.to(DEVICE)
    cross_entropy_loss.to(DEVICE)
    bce_loss.to(DEVICE)
    
    # ***********************       Summary Writer      ***********************
    # writer_G = SummaryWriter(os.path.join(LOGS))
    # writer_coarse = SummaryWriter(os.path.join(LOGS))
    # writer_D_LR = SummaryWriter(os.path.join(LOGS))
    # writer_D_SR = SummaryWriter(os.path.join(LOGS))
    
    # writer_G = SummaryWriter(LOGS)
    # writer_coarse = SummaryWriter(LOGS)
    # writer_D_LR = SummaryWriter(LOGS)
    # writer_D_SR = SummaryWriter(LOGS)
    writer = SummaryWriter(os.path.join(LOGS, 'g'))
    writer_d = SummaryWriter(os.path.join(LOGS, 'd'))
    
    count = int(len(trainloader) // SAVE_IMG)
    
    label_real = torch.ones(BATCH_SIZE, 1, 14, 14).cuda()
    label_fake = torch.zeros(BATCH_SIZE, 1, 14, 14).cuda()
    
    for epoch in range(start_epoch, EPOCHS):
        losses_G = AverageMeter()
        losses_coarse = AverageMeter()
        losses_D_LR = AverageMeter()
        losses_D_SR = AverageMeter()
        batch_time = AverageMeter()
        
        coarse.train()
        encoder.train()
        decoder.train()
        g_lr.train()
        
        prior_estimation.eval()
        backbone.eval()
        
        sche_G.step(epoch)
        sche_D_LR.step(epoch)
        sche_D_SR.step(epoch)
        sche_coarse.step(epoch)
        
        bar = Bar('Processing: ', max=len(trainloader))
        
        prev_time = time.time()
        end_time = time.time()
        
        for i, batch in enumerate(tqdm(trainloader)):
            lr_img = batch['lr_img']
            hr_img = batch['hr_img']
            scale = batch['scale']
            
            lr_img = lr_img.cuda()
            hr_img = hr_img.cuda()
            
            # train D
            y_coarse = coarse(lr_img)
            prior_feature, landmark_out, parsing_out = prior_estimation(y_coarse)
            y_sr = decoder(torch.cat((prior_feature, encoder(y_coarse)), dim=1))
            
            loss_sr = bce_loss(d_sr(y_sr), label_fake) + bce_loss(d_sr(hr_img), label_real) + bce_loss(d_sr(y_coarse), label_fake)
            optim_D_SR.zero_grad()
            loss_sr.backward()
            optim_D_SR.step()
            losses_D_SR.update(loss_sr.data.cpu().numpy(), BATCH_SIZE)
            
            y_lr = g_lr(hr_img, label2onehot(scale, dim=3).cuda())
            y_src, y_cls = d_lr(y_lr)
            out_src, out_cls = d_lr(lr_img)
            loss_lr = bce_loss(y_src, label_fake) + bce_loss(out_src, label_real)
            
            label_cls = torch.ones(BATCH_SIZE, ) * scale.float()
            label_cls = label_cls.long()
            loss_lr += F.cross_entropy(out_cls, label_cls.cuda())
            
            optim_D_LR.zero_grad()
            loss_lr.backward()
            optim_D_LR.step()
            losses_D_LR.update(loss_lr.data.cpu().numpy(), BATCH_SIZE)
            
            # train G
            if i % STEP_G == 0:
                # *************************   coarse   ***************************
                y_coarse = coarse(lr_img)
                y_lr = g_lr(hr_img, label2onehot(scale, dim=3).cuda())
                y_src, y_cls = d_lr(y_lr)

                loss_sr = bce_loss(d_sr(y_coarse), label_real)
                loss_lr = bce_loss(y_src, label_real)
                label_cls = torch.ones(BATCH_SIZE, ) * scale.float()
                label_cls = label_cls.long()
                loss_lr += F.cross_entropy(y_cls, label_cls.cuda())

                # hr-->lr
                recon_coarse = coarse(y_lr)

                hr_logits_dict = get_feature(imgs=hr_img, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                sr_logits_dict = get_feature(imgs=y_coarse, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                recon_loss = 0.
                for layer in extract_layer_list:
                    temp_loss = mse_loss(hr_logits_dict[layer], sr_logits_dict[layer])
                    recon_loss += LAMBDA_FEATURE * temp_loss

                recon_loss += LAMBDA_PIXEL * mse_loss(hr_img, recon_coarse)

                # lr-->sr
                recon_lr = g_lr(y_coarse, label2onehot(scale, dim=3).cuda())

                lr_logits_dict = get_feature(imgs=lr_img, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                recon_lr_logits_dict = get_feature(imgs=y_lr, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)

                for layer in extract_layer_list:
                    temp_loss = mse_loss(lr_logits_dict[layer], recon_lr_logits_dict[layer])
                    recon_loss += LAMBDA_FEATURE * temp_loss

                recon_loss += LAMBDA_PIXEL * mse_loss(lr_img, recon_lr)

                g_loss = loss_sr + loss_lr + recon_loss

                optim_coarse.zero_grad()
                g_loss.backward()
                optim_coarse.step()
                losses_coarse.update(g_loss.data.cpu().numpy(), BATCH_SIZE)
                
                # *************************   overall   ***************************
                if epoch >= WARMING_UP:
                    y_coarse = coarse(lr_img)
                    prior_feature, landmark_out, parsing_out = prior_estimation(y_coarse)
                    y_sr = decoder(torch.cat((prior_feature, encoder(y_coarse)), dim=1))
                    y_lr = g_lr(hr_img, label2onehot(scale, dim=3).cuda())
                    y_src, y_cls = d_lr(y_lr)
                    
                    loss_sr = bce_loss(d_sr(y_sr), label_real)
                    loss_lr = bce_loss(y_src, label_real)
                    label_cls = torch.ones(BATCH_SIZE, ) * scale.float()
                    label_cls = label_cls.long()
                    loss_lr += F.cross_entropy(y_cls, label_cls.cuda())
                    
                    # ****************************      coarse      ********************************
                    # hr-->lr
                    recon_coarse = coarse(y_lr)
                    
                    hr_logits_dict = get_feature(imgs=hr_img, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    sr_logits_dict = get_feature(imgs=y_coarse, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    recon_loss = 0.
                    for layer in extract_layer_list:
                        temp_loss = mse_loss(hr_logits_dict[layer], sr_logits_dict[layer])
                        recon_loss += LAMBDA_FEATURE * temp_loss
                    
                    recon_loss += LAMBDA_PIXEL * mse_loss(hr_img, recon_coarse)
                    
                    # lr-->sr
                    recon_lr = g_lr(y_coarse, label2onehot(scale, dim=3).cuda())
                    
                    # lr_logits_dict = get_feature(imgs=lr_img, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    # recon_lr_logits_dict = get_feature(imgs=y_lr, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    #
                    # for layer in extract_layer_list:
                    #     temp_loss = mse_loss(lr_logits_dict[layer], recon_lr_logits_dict[layer])
                    #     recon_loss += LAMBDA_FEATURE * temp_loss
                    
                    recon_loss += LAMBDA_PIXEL * mse_loss(lr_img, recon_lr)
                    
                    # *******************************       G       ***********************************
                    # hr-->lr
                    recon_sr = decoder(torch.cat((prior_estimation(coarse(y_lr))[0], encoder(coarse(y_lr))), dim=1))
                    
                    hr_logits_dict = get_feature(imgs=hr_img, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    sr_logits_dict = get_feature(imgs=y_sr, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    
                    # recon_loss = 0.
                    for layer in extract_layer_list:
                        temp_loss = mse_loss(hr_logits_dict[layer], sr_logits_dict[layer])
                        recon_loss += LAMBDA_FEATURE * temp_loss
                    
                    recon_loss += LAMBDA_PIXEL * mse_loss(hr_img, recon_sr)
                    
                    # lr-->sr-->lr
                    recon_lr = g_lr(y_sr, label2onehot(scale, dim=3).cuda())
                    
                    lr_logits_dict = get_feature(imgs=lr_img, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    recon_lr_logits_dict = get_feature(imgs=y_lr, backbone=backbone, backbone_name=BACKBONE_NAME, layer_list=layer_list)
                    
                    for layer in extract_layer_list:
                        temp_loss = mse_loss(lr_logits_dict[layer], recon_lr_logits_dict[layer])
                        recon_loss += LAMBDA_FEATURE * temp_loss
                    
                    recon_loss += LAMBDA_PIXEL * mse_loss(lr_img, recon_lr)
                    
                    g_loss = loss_sr + loss_lr + recon_loss
                    
                    optim_G.zero_grad()
                    g_loss.backward()
                    optim_G.step()
                    losses_G.update(g_loss.data.cpu().numpy(), BATCH_SIZE)
            
            # Determine approximate time left.
            batch_time.update(time.time() - end_time)
            end_time = time.time()
            batchs_done = epoch * len(trainloader) + i
            batchs_left = EPOCHS * len(trainloader) - batchs_done
            time_left = datetime.timedelta(seconds=batchs_left * (time.time() - prev_time))
            prev_time = time.time()
            
            bar.suffix = '  Epoch/Step: {epoch}/{step} | G: {g:.4f} | Coarse: {coarse:.4f} | D_SR: {d_sr:.4f} | ' \
                         'D_LR: {d_lr:.4f} | ETA: {time_left}'.format(
                epoch=epoch,
                step=i,
                g=losses_G.avg,
                coarse=losses_coarse.avg,
                d_sr=losses_D_SR.avg,
                d_lr=losses_D_LR.avg,
                time_left=time_left
            )
            print(bar.suffix)
            
            # save image
            if i % count == 0:
                img = lr_img.detach().cpu().numpy()[0]
                img_name = 'lr_img_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                img = y_coarse.detach().cpu().numpy()[0]
                img_name = 'coarse_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                img = hr_img.detach().cpu().numpy()[0]
                img_name = 'hr_img_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                img = y_lr.detach().cpu().numpy()[0]
                img_name = 'fake_lr_img_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                img = recon_lr.detach().cpu().numpy()[0]
                img_name = 'recon_lr_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                img = recon_coarse.detach().cpu().numpy()[0]
                img_name = 'recon_coarse_{}_{}.jpg'.format(epoch, i // count)
                image = m.toimage(img, cmin=None, cmax=None)
                image.save(os.path.join(RESULT, img_name))
                
                if epoch >= WARMING_UP:
                    img = recon_sr.detach().cpu().numpy()[0]
                    img_name = 'recon_sr_{}_{}.jpg'.format(epoch, i // count)
                    image = m.toimage(img, cmin=None, cmax=None)
                    image.save(os.path.join(RESULT, img_name))
                    
                    img = y_sr.detach().cpu().numpy()[0]
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
        
        # writer_G.add_scalar('g/g', losses_G.avg, epoch)
        # writer_coarse.add_scalar('g/coarse', losses_coarse.avg, epoch)
        # writer_D_LR.add_scalar('d/d_lr', losses_D_LR.avg, epoch)
        # writer_D_SR.add_scalar('d/d_sr', losses_D_SR.avg, epoch)
        
        writer.add_scalar('g/g', losses_G.avg, epoch)
        writer.add_scalar('g/coarse', losses_coarse.avg, epoch)
        writer_d.add_scalar('d/d_sr', losses_D_SR.avg, epoch)
        writer_d.add_scalar('d/d_lr', losses_D_LR.avg, epoch)
        
        if (epoch + 1) % SAVE_MODEL == 0:
            torch.save({'state_dict': coarse.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'coarse_' + FLAG + '.pth'))
            torch.save({'state_dict': encoder.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'encoder_' + FLAG + '.pth'))
            torch.save({'state_dict': decoder.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'decoder_' + FLAG + '.pth'))
            torch.save({'state_dict': d_lr.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'd_lr_' + FLAG + '.pth'))
            torch.save({'state_dict': d_sr.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'd_sr_' + FLAG + '.pth'))
            torch.save({'state_dict': g_lr.state_dict(), 'epoch': epoch}, os.path.join(CHECKPOINT, 'g_lr_' + FLAG + '.pth'))
            
            torch.save(optim_G.state_dict(), os.path.join(CHECKPOINT, 'optim_g_' + FLAG + '.pth'))
            torch.save(optim_D_LR.state_dict(), os.path.join(CHECKPOINT, 'optim_d_lr_' + FLAG + '.pth'))
            torch.save(optim_D_SR.state_dict(), os.path.join(CHECKPOINT, 'optim_d_sr_' + FLAG + '.pth'))
            torch.save(optim_coarse.state_dict(), os.path.join(CHECKPOINT, 'optim_coarse_' + FLAG + '.pth'))
        
        if (epoch + 1) % N_EVAL == 0:
            with torch.no_grad():
                coarse.eval()
                encoder.eval()
                decoder.eval()
                prior_estimation.eval()
                
                tf = torchvision.transforms.Compose(
                    [
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                )
                
                if not os.path.exists(RESULT_EVAL):
                    os.mkdir(RESULT_EVAL)
                
                # dir = 'scface'
                # img_list = os.listdir(os.path.join(EVAL_ROOT, dir))
                # if not os.path.exists(os.path.join(RESULT_EVAL, dir)):
                #     os.mkdir(os.path.join(RESULT_EVAL, dir))
                # for img_nam in tqdm(img_list, desc='Eval scface ...'):
                #     img = Image.open(os.path.join(EVAL_ROOT, dir, img_nam)).resize((112, 112)).convert('RGB')
                #     img = tf(img).unsqueeze(0).to(DEVICE)
                #     coarse_img = coarse(img)
                #     # *******       prior estimation        *******
                #     prior_feature, landmark_out, parsing_out = prior_estimation(coarse_img)
                #
                #     # *******       encoder & decoder     *******
                #     encoder_feature = encoder(coarse_img)
                #     feature = torch.cat((prior_feature, encoder_feature), dim=1)
                #     sr_img = decoder(feature)
                #
                #     img = img.detach().cpu().numpy()[0]
                #     img_name = 'lr_' + img_nam
                #     image = m.toimage(img, cmin=None, cmax=None)
                #     image.save(os.path.join(RESULT_EVAL, dir, img_name))
                #
                #     img = coarse_img.detach().cpu().numpy()[0]
                #     img_name = 'coarse_' + img_nam
                #     image = m.toimage(img, cmin=None, cmax=None)
                #     image.save(os.path.join(RESULT_EVAL, dir, img_name))
                #
                #     img = sr_img.detach().cpu().numpy()[0]
                #     img_name = 'sr_' + img_nam
                #     image = m.toimage(img, cmin=None, cmax=None)
                #     image.save(os.path.join(RESULT_EVAL, dir, img_name))
                #
                #     landmark_out = landmark_out.detach().cpu().numpy()[0]
                #     img = np.sum(landmark_out, axis=0)
                #     img_name = 'landmark_' + img_nam
                #     plt.axis('off')
                #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
                #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
                #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                #     plt.margins(0, 0)
                #     plt.imshow(img)
                #     plt.savefig(os.path.join(RESULT_EVAL, dir, img_name), transparent=True, pad_inches=0)
                #
                #     img = parsing_out[0].max(0)[1].detach().cpu().numpy()
                #     img = trainset.decode_segmap(img)
                #     img_name = 'parsing_' + img_nam
                #
                #     plt.imsave(os.path.join(RESULT_EVAL, dir, img_name), img)

                dir = 'FaceSurv'
                img_list = os.listdir(os.path.join(EVAL_ROOT, dir))
                if not os.path.exists(os.path.join(RESULT_EVAL, dir)):
                    os.mkdir(os.path.join(RESULT_EVAL, dir))
                for img_nam in tqdm(img_list, desc='Eval FaceSurv ...'):
                    img = Image.open(os.path.join(EVAL_ROOT, dir, img_nam)).resize((112, 112)).convert('RGB')
                    img = tf(img).unsqueeze(0).to(DEVICE)
                    coarse_img = coarse(img)
                    # *******       prior estimation        *******
                    prior_feature, landmark_out, parsing_out = prior_estimation(coarse_img)
    
                    # *******       encoder & decoder     *******
                    encoder_feature = encoder(coarse_img)
                    feature = torch.cat((prior_feature, encoder_feature), dim=1)
                    sr_img = decoder(feature)
    
                    img = img.detach().cpu().numpy()[0]
                    img_name = 'lr_' + img_nam
                    image = m.toimage(img, cmin=None, cmax=None)
                    image.save(os.path.join(RESULT_EVAL, dir, img_name))
    
                    img = coarse_img.detach().cpu().numpy()[0]
                    img_name = 'coarse_' + img_nam
                    image = m.toimage(img, cmin=None, cmax=None)
                    image.save(os.path.join(RESULT_EVAL, dir, img_name))
    
                    img = sr_img.detach().cpu().numpy()[0]
                    img_name = 'sr_' + img_nam
                    image = m.toimage(img, cmin=None, cmax=None)
                    image.save(os.path.join(RESULT_EVAL, dir, img_name))
    
                    landmark_out = landmark_out.detach().cpu().numpy()[0]
                    img = np.sum(landmark_out, axis=0)
                    img_name = 'landmark_' + img_nam
                    plt.axis('off')
                    plt.gca().xaxis.set_major_locator(plt.NullLocator())
                    plt.gca().yaxis.set_major_locator(plt.NullLocator())
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.margins(0, 0)
                    plt.imshow(img)
                    plt.savefig(os.path.join(RESULT_EVAL, dir, img_name), transparent=True, pad_inches=0)
    
                    img = parsing_out[0].max(0)[1].detach().cpu().numpy()
                    img = trainset.decode_segmap(img)
                    img_name = 'parsing_' + img_nam
    
                    plt.imsave(os.path.join(RESULT_EVAL, dir, img_name), img)


if __name__ == '__main__':
    main()
