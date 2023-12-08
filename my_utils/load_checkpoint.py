import torch
from my_utils.events import LOGGER
from my_utils.torch_utils import fuse_model


def load_checkpoint(weights, map_location=None, inplace=True, fuse=True):
    '''Load model from checkpoint file. '''
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    if fuse:
        LOGGER.info("\nFusing model...")
        model = fuse_model(model).eval()
    else:
        model = model.eval()
    return model
