from .unet import UNet
from .asf_unet import ASFUNet
from hydra.utils import instantiate


def get_decoder_by_cfg(config):
    decoder = instantiate(config)
    return decoder
