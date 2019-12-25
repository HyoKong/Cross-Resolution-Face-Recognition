from __future__ import print_function
import os
import logging
import time
import torch.nn as nn

def init_log(output_dir,log = 'log.log'):
    logging.basicConfig(level=logging.CRITICAL,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, log),
                        filemode='a')
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)
    
    return logging

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        # self.submodule = submodule
        # self.extracted_layers = extracted_layers

    def forward(self, x, extracted_layers, submodule):
        outputs = {}
        times = {}
        start = time.time()
        for name, module in submodule._modules.items():
            x = module(x)
            temp = time.time() - start
            if name in extracted_layers:
                outputs[name] = x
                times[name] = temp
        return outputs, times, x, temp

if __name__ == '__main__':
    pass
