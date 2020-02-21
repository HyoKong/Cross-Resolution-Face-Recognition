import torch

SEED = 666
SAVE_IMG = 1
SAVE_MODEL = 1
FLAG = 'facesurf'
N_EVAL = 10

BACKBONE_NAME = 'IR50'
if BACKBONE_NAME == 'lightcnn':
    BACKBONE_RESUME_ROOT = '/home/hyo/Projects/GAN/Cross-Resolution-Face-Recognition/SUPER_RESOLUTION/checkpoint/LightCNN_29Layers_V2_checkpoint.pth.tar'
else:
    BACKBONE_RESUME_ROOT = '/home/hyo/Projects/GAN/Cross-Resolution-Face-Recognition/SUPER_RESOLUTION/checkpoint/backbone_ir50_ms1m_epoch120.pth'
    
# IMG_ROOT = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/scface'
# IMG_ROOT = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/lfw_aligned'
# SR_IMG_ROOT = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/scface/gallery'
IMG_ROOT = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/FaceSurv/FaceSurv_DayData [Annotations]_aligned'
SR_IMG_ROOT = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/FaceSurv/Gallery_aligned'
OPENSET_PATH = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/FaceSurv/Training and Testing Split/componentWiseDetails.txt'
RESULT = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/result_facesurf'
RESULT_EVAL = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/result_eval_GAN'
EVAL_ROOT = '/media/hyo/文档/Cross_Resolution_Dataset/CelebAMask-HQ-CR/CelebAMask-HQ-CR/eval'

WARMING_UP = 0
EPOCHS = 200 + WARMING_UP
BATCH_SIZE = 3
NUM_WORKERS = 8
MILESTONES = [50 + WARMING_UP, 100 + WARMING_UP, 150 + WARMING_UP]
STEP_G = 2

LR_COARSE = 0.001
LR_PRIOR = 0.01 * 0.01
LR_ENCODER_DECODER = 0.01
LR_G = 0.001
LR_D = 0.0005
WD = 1e-5
BETA1 = 0.5
BETA2 = 0.999
D_BETA1 = 0.5
D_BETA2 = 0.999

LAMBDA_PIXEL = 50
LAMBDA_FEATURE = 1
LAMBDA_PARSING = 1
LAMBDA_LANDMARK = 1

RESUME_COARSE = ''
RESUME_PRIOR_ESTIMATION = 'checkpoint/prior_estimation_' + 'backbone_L' + '.pth'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

CHECKPOINT = './checkpoint_facesurf'
LOGS = './logs_facesurf'