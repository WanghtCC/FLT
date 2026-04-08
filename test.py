import os.path
import math
import torch
import lpips
import random
import logging
import argparse
import numpy as np
from data import data_manager
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from models.select_model import define_Model

def load_network(self, load_path, network, strict=True, param_key='params'):
    network = self.get_bare_model(network)
    if strict:
        state_dict = torch.load(load_path)
        if param_key in state_dict.keys():
            state_dict = state_dict[param_key]
        network.load_state_dict(state_dict, strict=strict)
    else:
        state_dict_old = torch.load(load_path)
        if param_key in state_dict_old.keys():
            state_dict_old = state_dict_old[param_key]
        state_dict = network.state_dict()
        for ((key_old, param_old),(key, param)) in zip(state_dict_old.items(), state_dict.items()):
            state_dict[key] = param_old
        network.load_state_dict(state_dict, strict=True)
        del state_dict_old, state_dict

def main(json_path='options/train_swinir_sr.json'):
    parser = argparse.ArgumentParser(description='USSR Test-100')
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # seed
    # ----------------------------------------
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print('Random Seed: {}'.format(seed))

    '''
    ------------------------------------
    Step-2: data preparation
    ------------------------------------
    '''
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = data_manager.init_dataset(dataset_opt, None)
            test_loader = DataLoader(test_set,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=1,
                                     drop_last=False,
                                     pin_memory=True)

    print(test_set.__len__())
    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    from models.networks.network_flt import SwinSFR as net
    model = net(upscale=opt['netG']['upscale'],
               in_chans=opt['netG']['in_chans'],
               img_size=opt['netG']['img_size'],
               window_size=opt['netG']['window_size'],
               img_range=opt['netG']['img_range'],
               depths=opt['netG']['depths'],
               embed_dim=opt['netG']['embed_dim'],
               num_heads=opt['netG']['num_heads'],
               mlp_ratio=opt['netG']['mlp_ratio'],
               upsampler=opt['netG']['upsampler'],
               resi_connection=opt['netG']['resi_connection'])
    
    print(f"{model.flops() / 1e9} GFLOPs")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6} M")

    load_path = opt['path']['pretrained_netG']
    print('Loading model for G [{:s}] ...'.format(load_path))
    state_dict = torch.load(load_path)
    if 'params' in state_dict.keys():
        state_dict = state_dict['params']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    lpips_loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_loss_fn = lpips_loss_fn.cuda()

    idx = 0
    for test_data in test_loader:
        idx += 1
        image_name_ext = os.path.basename(test_data['L_path'][0])
        img_name, ext = os.path.splitext(image_name_ext)
        img_dir_L = os.path.join(opt['path']['images'], 'test_images', 'L')
        img_dir_E = os.path.join(opt['path']['images'], 'test_images', 'E')
        img_dir_H = os.path.join(opt['path']['images'], 'test_images', 'H')
        if not os.path.exists(img_dir_L):
            util.mkdir(img_dir_L)
        if not os.path.exists(img_dir_E):
            util.mkdir(img_dir_E)
        if not os.path.exists(img_dir_H):
            util.mkdir(img_dir_H)
        img_L = test_data['L']
        img_H = test_data['H']
        with torch.no_grad():
            img_E = model(img_L)
        
        img_L = img_L.detach()[0].float()
        img_E = img_E.detach()[0].float()
        img_H = img_H.detach()[0].float()
        
        L_img = util.tensor2uint(img_L)
        E_img = util.tensor2uint(img_E)
        H_img = util.tensor2uint(img_H)
        save_img_path = os.path.join(img_dir_L, '{:s}.png'.format(img_name))
        util.imsave(L_img, save_img_path)
        save_img_path = os.path.join(img_dir_E, '{:s}.png'.format(img_name))
        util.imsave(E_img, save_img_path)
        save_img_path = os.path.join(img_dir_H, '{:s}.png'.format(img_name))
        util.imsave(H_img, save_img_path)
        print(save_img_path)


if __name__ == '__main__':
    main()