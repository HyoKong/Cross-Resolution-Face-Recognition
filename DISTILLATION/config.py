import torch

SEED = 666
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

INPUT_SIZE = [112, 112]  # support: [112, 112] and [224, 224]
RGB_MEAN = [0.5, 0.5, 0.5]  # for normalize inputs to [-1, 1]
RGB_STD = [0.5, 0.5, 0.5]
EMBEDDING_SIZE = 512  # feature dimension
LAYER_LIST = ['2', '6', '21', '24']


EPOCHS = 120
BATCH_SIZE = 1
PIN_MEMORY = True
NUM_WORKERS = 8
DROP_LAST = True
WEIGHT_DECAY = 1e-5
LR = 0.1
MOMENTUM = 0.9
MILESTONES = [30, 60, 90]

DATA_ROOT = r'D:\Hyo_Dataset\FaceDataset'  # the parent root where your train/val/test data are stored
BACKBONE_RESUME_ROOT = 'DISTILLATION/checkpoint/IR/backbone_ir50_ms1m_epoch120.pth'

CHECKPOINT = 'DISTILLATION/checkpoint'

RESUME_ROOT = ''
