import os
import math
import time
import random
import shutil
import pdb
from torch.nn.init import xavier_uniform as xavier
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.checkpoint import checkpoint_sequential,checkpoint
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from model.FSRnet import *
import argparse

from utils import Bar,Logger,AverageMeter,normalizedME,mkdir_p,savefig,visualize
from utils.utils import schedule_lr
from helen_loader import *
from loss.loss import MSELossFunc,CrossEntropyLoss2d,MSELoss_Landmark,MMD

def main():
    parser = argparse.ArgumentParser("FSR Network on pytorch")
    
    # Datasets
    parser.add_argument('--dataset_dir', type=str, default="")
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default = 4)')
    
    # Optimization option
    parser.add_argument('--epochs', default=3000, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--gan_epochs',default=2,type=int,help='number of epochs for training discriminator')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful to restart)')
    parser.add_argument('--train_batch', type=int, default=3, help='train batch size', metavar='N')
    parser.add_argument('--test_batch', type=int, default=3, help='test batch size', metavar='N')
    parser.add_argument('--lr', '--learning_rate', default=0.01*0.5, type=float, help='initial learning rate',
                        metavar='LR')
    parser.add_argument('--lr_decay_rate', default=0.9993, type=float)
    parser.add_argument('--drop', '--dropout', default=0.0, type=float, metavar='Dropout', help='Dropout ratio')
    parser.add_argument('--schedule', type=int, default=[100, 300], nargs='+',
                        help='Decrease learning rate at these epochs')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on shedule')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float, metavar='W', help='weight decay')
    
    # Checkpoint
    parser.add_argument('-c', '--checkpoint', default='./checkpoint/', type=str, metavar='PATH',
                        help='Path to save checkpoint')
    # parser.add_argument('--resume', default='', type=str, metavar='PATH',help='path to ltest checkpoint (default=None)')
    parser.add_argument('--resume',default = './checkpoint/FSRGAN_28_antialias.pth.tar',type=str)
    
    # Miscs
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='eval model on validation set')
    
    # Device option
    parser.add_argument('--gpu_id', type=str, default='0')
    args = parser.parse_args()

    print(args)

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.environ['CUDA_ENABLE_DEVICES'] = '0'
    torch.cuda.set_device(0)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("torch.backends.cudnn.version: {}".format(torch.backends.cudnn.version()))

    # Random seed
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.manualSeed)

    global best_loss
    best_loss = 9999999

    # Data
    print('====> Preparing dataset <====')
    trainset = HelenLoader(is_transform=True, split='all',train_mode=True)
    trainloader = data.DataLoader(trainset, batch_size=args.train_batch, shuffle=True, num_workers=args.workers)

    testset = HelenLoader(is_transform=True, split='test_no_rotate',train_mode=False)
    testloader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=True, num_workers=args.workers)

    # model
    model = OverallNetwork_GAN()
    # model = Course_SR_Network()
    # model = Prior_Estimation_Network()
    model.apply(weights_init)
    cudnn.benchmark = True
    print('Total params:  %.2fM' % (sum(p.numel() for p in model.parameters()) / 100000.0))
    
    start_epoch = 0
    title = 'FSRGAN'

     
    #  # network parameters
    # coarse_params = list(map(id,model._coarse_sr_network.parameters()))
    # prior_params = list(map(id,model._prior_estimation_network.parameters()))
    # encoder_params = list(map(id,model._fine_sr_encoder.parameters()))
    # decoder_params = list(map(id,model._fine_sr_decoder.parameters()))
    # discriminator_params = list(map(id,model._discriminator.parameters()))
    # base_params = filter(lambda p:id(p) not in discriminator_params ,model.parameters())
    
    # criterion
    criterion_mse = MSELossFunc().cuda()
    criterion_cross_entropy = CrossEntropyLoss2d().cuda()
    criterion_landmark = MSELoss_Landmark().cuda()
    criterion_mmd = MMD().cuda()
    
    if use_cuda:
        model = model.cuda()

    coarse_optim = torch.optim.RMSprop(model._coarse_sr_network.parameters(), lr=args.lr, weight_decay=args.weight_decay,alpha=0.99)
    prior_optim = torch.optim.RMSprop(model._prior_estimation_network.parameters(), lr=args.lr,weight_decay=args.weight_decay, alpha=0.99)
    encoder_optim = torch.optim.RMSprop(model._fine_sr_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay,alpha=0.99)
    decoder_optim = torch.optim.RMSprop(model._fine_sr_decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay,alpha=0.99)
    discriminator_optim = torch.optim.RMSprop(model._discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay,alpha=0.99)
    # optimizer = torch.optim.RMSprop([{'params':base_params,'lr':lr,'weight_decay':args.weight_decay,'alpha':0.99}],lr=lr,weight_decay=args.weight_decay)
    # overall_optim = torch.optim.RMSprop(model.parameters(),lr=lr,weight_decay=args.weight_decay)
    optimizer = None
    overall_optim = None
    
    # pdb.set_trace()

    # load model
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        coarse_optim.load_state_dict(checkpoint['coarse_optim'])
        encoder_optim.load_state_dict(checkpoint['encoder_optim'])
        decoder_optim.load_state_dict(checkpoint['decoder_optim'])
        prior_optim.load_state_dict(checkpoint['prior_optim'])
        discriminator_optim.load_state_dict(checkpoint['discriminator_optim'])
        optimizer = None
        overall_optim = None

    
    
    for epoch in range(start_epoch,args.epochs):
        lr = args.lr
        # if epoch <10:
        #     lr = args.lr
        # elif epoch <20:
        #     lr = args.lr*0.5
        # elif epoch <30:
        #     lr = args.lr*0.1
        # elif epoch < 40:
        #     lr = args.lr*0.05
        # else:
        #     lr = args.lr*0.01
        if epoch > 100:
            schedule_lr(coarse_optim,args.lr*0.1*2.2,args.lr_decay_rate,epoch-100)
            schedule_lr(prior_optim,args.lr*0.05,args.lr_decay_rate,epoch-100)
            schedule_lr(encoder_optim,args.lr*0.1*1.5,args.lr_decay_rate,epoch-100)
            schedule_lr(decoder_optim,args.lr*0.1*1.5,args.lr_decay_rate,epoch-100)
            schedule_lr(discriminator_optim,args.lr*0.04,args.lr_decay_rate,epoch-100)
            lr = args.lr*0.1 * args.lr_decay_rate ** (epoch-100)
        elif epoch>50:
            schedule_lr(coarse_optim, args.lr*0.5, 1, epoch)
            schedule_lr(prior_optim, args.lr*0.5, 1, epoch)
            schedule_lr(encoder_optim, args.lr*0.5, 1, epoch)
            schedule_lr(decoder_optim, args.lr*0.5, 1, epoch)
            schedule_lr(discriminator_optim, args.lr*0.5, 1, epoch)
            lr = args.lr*0.5
        else:
            schedule_lr(coarse_optim, args.lr, 1, epoch)
            schedule_lr(prior_optim, args.lr, 1, epoch )
            schedule_lr(encoder_optim, args.lr, 1, epoch)
            schedule_lr(decoder_optim, args.lr, 1, epoch)
            schedule_lr(discriminator_optim, args.lr, 1, epoch)
            lr = args.lr

        # if args.resume:
        #
        #     coarse_optim = checkpoint['coarse_optim']['param_groups']
        #     prior_optim = checkpoint['prior_optim']
        #     encoder_optim = checkpoint['encoder_optim']
        #     decoder_optim = checkpoint['decoder_optim']
        #     discriminator_optim = checkpoint['discriminator_optim']['state']
        
        print('\nEpoch: [%d | %d] LR: %.8f' % (epoch + 1, args.epochs, lr))
        train_loss = train(trainloader,model,criterion_mse,criterion_cross_entropy,criterion_landmark,
                           criterion_mmd,overall_optim,coarse_optim,prior_optim,encoder_optim,decoder_optim, discriminator_optim,
                           epoch,args.gan_epochs,use_cuda,train_batch=args.train_batch,lr=lr)
        test_loss = test(testloader, model, criterion_mse, criterion_cross_entropy, criterion_landmark,
             criterion_mmd, optimizer, epoch, use_cuda, test_batch=args.test_batch)

        is_best = best_loss > test_loss
        best_loss = min(test_loss, best_loss)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'coarse_optim': coarse_optim.state_dict(),
            'prior_optim': prior_optim.state_dict(),
            'encoder_optim': encoder_optim.state_dict(),
            'decoder_optim': decoder_optim.state_dict(),
            'discriminator_optim':discriminator_optim.state_dict(),
        }, is_best, checkpoint=args.checkpoint, filename=title + '_28_antialias' + '.pth.tar')

def train(train_loader, model, criterion_mse, criterion_cross_entropy,criterion_landmark,criterion_mmd,
          overall_optim,coarse_optim,prior_optim,encoder_optim,decoder_optim, discriminator_optim,
          epoch,gan_epochs, use_cuda,train_batch,lr):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    gan_losses = AverageMeter()
    coarse_losses = AverageMeter()
    prior_losses = AverageMeter()
    encoder_losses = AverageMeter()
    decoder_losses = AverageMeter()
    
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    count = 0
    
    for batch_idx, [batch_lr_img, batch_sr_img, batch_lbl, batch_landmark] in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # pdb.set_trace()
        if use_cuda:
            batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = batch_lr_img.cuda(), batch_sr_img.cuda(), batch_lbl.cuda(), batch_landmark.cuda()
        batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = Variable(batch_lr_img), Variable(batch_sr_img), \
                                                                Variable(batch_lbl), Variable(batch_landmark)
        
        # for i in range(gan_epochs-1):
        #     # coarse_out, our_sr, out_landmark, out_lbl, out_embedding1 = checkpoint(model,2,batch_lr_img)
        #     # pdb.set_trace()
        #     # coarse_out, our_sr, out_landmark, out_lbl, out_embedding1 = model(batch_lr_img)
        #     # coarse_out, our_sr, out_landmark, out_lbl, out_embedding2 = model(batch_sr_img)
        #     # coarse_out1, our_sr1, out_landmark1, out_lbl1, out_embedding1, coarse_out2, our_sr2, out_landmark2, out_lbl2, out_embedding2 = model(batch_lr_img,batch_sr_img)
        #     out_sr,out_coarse,out_landmark,out_lbl,out_embedding1,out_embedding2 = model(batch_lr_img,batch_sr_img)
        #     gan_loss = -criterion_mmd(out_embedding1,out_embedding2)
        #     gan_losses.update(gan_loss.data.cpu().numpy(),batch_lr_img.size(0))
        #     discriminator_optim.zero_grad()
        #     gan_loss.backward()
        #     discriminator_optim.step()
            
        
        
        out_sr, out_coarse, out_landmark, out_lbl, out_embedding1, out_embedding2 = model(batch_lr_img, batch_sr_img)
        gan_loss = -criterion_mmd(out_embedding1, out_embedding2)
        gan_losses.update(gan_loss.data.cpu().numpy(), batch_lr_img.size(0))
        discriminator_optim.zero_grad()
        gan_loss.backward(retain_graph=True)
        discriminator_optim.step()
        
        coarse_loss = 12. * criterion_mse(out_coarse, batch_sr_img)
        coarse_losses.update(coarse_loss.data.cpu().numpy())
        coarse_optim.zero_grad()
        coarse_loss.backward(retain_graph=True)
        coarse_optim.step()
        
        encoder_loss = 10.0 * criterion_mse(out_sr, batch_sr_img) - gan_loss
        encoder_losses.update(encoder_loss.data.cpu().numpy())
        encoder_optim.zero_grad()
        encoder_loss.backward(retain_graph=True)
        encoder_optim.step()
        
        prior_loss = -gan_loss + criterion_mse(out_sr, batch_sr_img) + criterion_landmark(out_landmark,batch_landmark) + 1.0 * criterion_cross_entropy(out_lbl, batch_lbl)
        prior_losses.update(prior_loss.data.cpu().numpy())
        prior_optim.zero_grad()
        prior_loss.backward(retain_graph=True)
        prior_optim.step()
        
        decoder_loss = 10. * criterion_mse(out_sr, batch_sr_img)
        decoder_losses.update(decoder_loss.data.cpu().numpy())
        decoder_optim.zero_grad()
        decoder_loss.backward()
        decoder_optim.step()
        
        loss = (8. * criterion_mse(out_sr, batch_sr_img) + 8. * criterion_mse(out_coarse, batch_sr_img)
                + criterion_landmark(out_landmark, batch_landmark) + criterion_cross_entropy(out_lbl, batch_lbl)
                +criterion_mmd(out_embedding1,out_embedding2)
                ) / (2.0 * train_batch)
        loss = loss.data.cpu().numpy()
        losses.update(loss,batch_lr_img.size(0))
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # plot progress
        bar.suffix = '(Epoch: {epoch} | Learning Rate: {lr:.8f} | {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total Loss: {loss:.6f} | ' \
                     'Gan Loss: {gan_loss:.6f} | Coarse Loss: {coarse_loss:.6f} | Encoder Loss: {encoder_loss:.6f} | Decoder Loss: {decoder_loss:.6f} | Prior Loss: {prior_loss:.6f}'.format(
            epoch=epoch,
            lr=lr,
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
            gan_loss=gan_losses.avg,
            coarse_loss=coarse_losses.avg,
            encoder_loss = encoder_losses.avg,
            decoder_loss = decoder_losses.avg,
            prior_loss = prior_losses.avg,
        )
        print(bar.suffix)

        count += 1

        if count % 300 == 0:
            ## count = 0
            # rand_id = random.randint(0, 4)
            random_img, random_landmark, random_parsing, random_coarse, sr_img, lr_img = \
                out_sr[0], out_landmark[0], out_lbl[0], out_coarse[0], batch_sr_img[0], batch_lr_img[0]
            ## pdb.set_trace()
            random_img, random_landmark, random_parsing, random_coarse = random_img.detach().cpu().numpy(), random_landmark.detach().cpu().numpy(), \
                                                                         random_parsing.max(dim=0)[
                                                                             1].detach().cpu().numpy(), random_coarse.detach().cpu().numpy()
            sr_img = sr_img.detach().cpu().numpy()
            lr_img = lr_img.detach().cpu().numpy()
            ## pdb.set_trace()
            visualize.save_image(random_coarse, random_img, random_landmark, random_parsing, lr_img, sr_img, epoch,
                                 if_train=True, count=int(count / 300))
            ##-----------------------------------------------------------------------
            # visualize coarse network
            # random_coarse = coarse_out[0]
            # random_landmark = landmark_out[0]
            # random_parsing = parsing_out[0]
            # #random_coarse = random_coarse.detach().cpu().numpy()
            # random_landmark = random_landmark.detach().cpu().numpy()
            # random_parsing = random_parsing.max(dim=0)[1].detach().cpu().numpy()
            # visualize.save_image(landmark=random_landmark,parsing=random_parsing,epoch=epoch,if_train=True,count=int(count/100))
        bar.next()

    bar.finish()
    return losses.avg


def test(test_loader, model, criterion_mse, criterion_cross_entropy, criterion_landmark,
         criterion_mmd, optimizer, epoch, use_cuda,test_batch):
    global best_loss
    
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    
    end = time.time()
    bar = Bar('Processing', max=len(test_loader))
    count = 0
    
    for batch_idx, [batch_lr_img, batch_sr_img, batch_lbl, batch_landmark] in enumerate(test_loader):
        with torch.no_grad():
            
            # measure data loading time
            data_time.update(time.time() - end)
            
            if use_cuda:
                batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = batch_lr_img.cuda(), batch_sr_img.cuda(), batch_lbl.cuda(), batch_landmark.cuda()
            batch_lr_img, batch_sr_img, batch_lbl, batch_landmark = Variable(batch_lr_img), Variable(
                batch_sr_img), Variable(batch_lbl), Variable(batch_landmark)
            
            # compute output
            
            out_sr,out_coarse,out_landmark,out_lbl,out_embedding1,out_embedding2 = model(batch_lr_img,batch_sr_img)
            loss = (8. * criterion_mse(out_sr, batch_sr_img) + 8. * criterion_mse(out_coarse, batch_sr_img)
                    + criterion_landmark(out_landmark, batch_landmark) + criterion_cross_entropy(out_lbl, batch_lbl)
                    + criterion_mmd(out_embedding1, out_embedding2)) / (2.0 * test_batch)
            loss = loss.data.cpu().numpy()
            losses.update(loss, batch_lr_img.size(0))
            
            count += 1
            if count % 100 == 0:
                # rand_id = random.randint(0, 4)
                ## count = 0
                random_img, random_landmark, random_parsing, random_coarse, sr_img, lr_img = out_sr[0], out_landmark[0], \
                                                                                             out_lbl[0], out_coarse[0], \
                                                                                             batch_sr_img[0], \
                                                                                             batch_lr_img[0]
                random_img, random_landmark, random_parsing, random_coarse = random_img.detach().cpu().numpy(), random_landmark.detach().cpu().numpy(), \
                                                                             random_parsing.max(dim=0)[
                                                                                 1].detach().cpu().numpy(), random_coarse.detach().cpu().numpy()
                sr_img = sr_img.detach().cpu().numpy()
                lr_img = lr_img.detach().cpu().numpy()
                
                visualize.save_image(random_coarse, random_img, random_landmark, random_parsing, lr_img, sr_img, epoch,
                                     if_train=False, count=int(count / 100))
                ##-----------------------------------------------------------------------
                ##visualize coarse network
                # random_coarse = coarse_out[0]
                # random_coarse = random_coarse.detach().cpu().numpy()
                # visualize.save_image(coarse_image=random_coarse,epoch=epoch,if_train=True,count=int(count/90))
                
                # random_landmark = landmark_out[0]
                # random_parsing = parsing_out[0]
                # random_coarse = random_coarse[0].detach().cpu().numpy()
                # random_landmark = random_landmark.detach().cpu().numpy()
                # random_parsing = random_parsing.max(dim=0)[1].detach().cpu().numpy()
                # visualize.save_image(landmark=random_landmark,parsing=random_parsing,epoch=epoch,if_train=False,count=int(count/5))
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
        
        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Loss: {loss:.6f}'.format(
            batch=batch_idx + 1,
            size=len(test_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            loss=losses.avg,
        )
        print(bar.suffix)
        bar.next()
    
    bar.finish()
    return losses.avg

def weights_init(m):
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        xavier(m.weight.data)
        xavier(m.bias.data)
    '''
    for each in m.modules():
        if isinstance(each,nn.Conv2d):
            nn.init.xavier_uniform_(each.weight.data)
            if each.bias is not None:
                each.bias.data.zero_()
        elif isinstance(each,nn.BatchNorm2d):
            each.weight.data.fill_(1)
            each.bias.data.zero_()
        # elif isinstance(each,nn.InstanceNorm2d):
        #     each.weight.data.fill_(1)
        #     each.bias.data.zero_()
        elif isinstance(each,nn.Linear):
            nn.init.xavier_uniform_(each.weight.data)
            each.bias.data.zero_()
            
def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

if __name__ == "__main__":
    main()