import hydra
from omegaconf import DictConfig

import colat.runner as runner
import colat.utils.log_utils as utils
from colat.classifier import *
from diffae.templates import *



@hydra.main(config_path="conf", config_name="gen_custom")
def gen_custom_golgi(cfg: DictConfig):
    conf = golgi_autoenc()
    cfg.diffae_ckpt_path = "/projects/deepdevpath2/Saranga/diffae_experiments/diffae/checkpoints/golgi_autoenc/last.ckpt"
    classifier = GolgiClassifier()
    checkpoint_name = "/projects/deepdevpath2/Saranga/DiffaeCLR_golgi/colat/Golgi_classifier.pth"
    classifier.load_state_dict(torch.load(checkpoint_name, map_location = runner.get_device(cfg)))
    for param in classifier.parameters():
            param.requires_grad = False
    classifier.eval()

    utils.display_config(cfg)
    runner.generate_custom(cfg, conf, classifier)


@hydra.main(config_path="conf", config_name="gen_custom")
def gen_custom_bbc(cfg: DictConfig):
    conf = bbc_autoenc()
    cfg.diffae_ckpt_path = "/projects/deepdevpath2/Saranga/diffae_experiments/diffae/checkpoints/bbc_autoenc/last.ckpt"
    classifier = BBCClassifier()
    checkpoint_name = "/projects/deepdevpath2/Saranga/diffae_experiments/diffaeCLR/Classifiers/BBC_classifier.pth"
    classifier.load_state_dict(torch.load(checkpoint_name, map_location = runner.get_device(cfg)))
    for param in classifier.parameters():
            param.requires_grad = False
    classifier.eval()

    utils.display_config(cfg)
    runner.generate_custom(cfg, conf, classifier)


@hydra.main(config_path="conf", config_name="gen_custom")
def gen_custom_larkk(cfg: DictConfig):
    # TODO : train LARKK classifier and update this
    conf = larkk_autoenc()
    cfg.diffae_ckpt_path = "/projects/deepdevpath2/Saranga/diffae_experiments/diffae/checkpoints/larkk_autoenc/last.ckpt"
    classifier = GolgiClassifier()
    checkpoint_name = "/projects/deepdevpath2/Saranga/DiffaeCLR_golgi/colat/Golgi_classifier.pth"
    classifier.load_state_dict(torch.load(checkpoint_name, map_location = runner.get_device(cfg)))
    for param in classifier.parameters():
            param.requires_grad = False
    classifier.eval()

    utils.display_config(cfg)
    runner.generate_custom(cfg, conf, classifier)


if __name__ == "__main__":
    gen_custom_golgi()
    # gen_custom_bbc()
    # gen_custom_larkk()

