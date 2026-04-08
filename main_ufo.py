from glob import glob
import os.path
import math
import cv2
import torch
import lpips
import random
import logging
import argparse
import numpy as np
from PIL import Image
from data import data_manager
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils import uiqm_utils as uiqm
from utils.utils_dist import get_dist_info, init_dist
from utils.imqual_utils import getSSIM, getPSNR

from models.select_model import define_Model

'''
------------------------------------
training code
------------------------------------
Haitao Wang
------------------------------------
Cite:
    https://github.com/cszn/KAIR
    https://github.com/xinntao/BasicSR
------------------------------------
'''
# torch.set_num_threads(1)    # Used to manually adjust the pytorch thread pool, because some hardware environments cannot adapt to the multi-threaded work of DataLoader.
# os.environ["TORCH_USE_FBGEMM"] = "0"    # Used to block FBGEM, because macOS cannot use this class library. If not macOS, please annotate the line code.
# print(torch.backends.mps.is_available())  # True means available. If not macOS, please annotate the line code.
# print(torch.backends.mps.is_built())     # True means that PyTorch compilation supports MPS. If not macOS, please annotate the line code.

def main(json_path='options/train_sr_ufo.json'):
    '''
    ------------------------------------
    Step-1: prepare opt
    ------------------------------------
    '''
    parser = argparse.ArgumentParser(description='Train sr model with PSNR loss')
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist', default=False)
    parser.add_argument('--print', default=True, help='print opt/model or not')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()

    if opt['rank'] == 0:
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')
    opt['path']['pretrained_netG'] = init_path_G
    opt['path']['pretrained_netE'] = init_path_E
    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    if opt['path']['pretrained_netG'] is None or opt['path']['pretrained_netE'] is None:
        opt['print'] = False
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)

    # ----------------------------------------
    # configure logger
    # ----------------------------------------
    if opt['rank'] == 0:
        logger_name = 'train'
        utils_logger.logger_info(logger_name, os.path.join(opt['path']['log'], logger_name+'.log'))
        logger = logging.getLogger(logger_name)
        if opt['print']:
            logger.info(option.dict2str(opt))

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
    logger.info('Random Seed: {}'.format(seed))
    '''
    ------------------------------------
    Step-2: data preparation
    ------------------------------------
    '''
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = data_manager.init_dataset(dataset_opt, logger=logger)
            train_size = int(math.ceil(len(train_set) / dataset_opt['dataloader_batch_size']))
            if opt['rank'] == 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            if opt['dist']:
                train_sampler = DistributedSampler(train_set, shuffle=dataset_opt['dataloader_shuffle'], drop_last=True, seed=seed)
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size']//opt['num_gpu'],
                                          shuffle=False,
                                          num_workers=dataset_opt['dataloader_num_workers']//opt['num_gpu'],
                                          drop_last=True,
                                          pin_memory=True,
                                          sampler=train_sampler)
            else:
                train_loader = DataLoader(train_set,
                                          batch_size=dataset_opt['dataloader_batch_size'],
                                          shuffle=dataset_opt['dataloader_shuffle'],
                                          num_workers=dataset_opt['dataloader_num_workers'],
                                          drop_last=True,
                                          pin_memory=True)

        elif phase == 'test':
            test_set = data_manager.init_dataset(dataset_opt, logger=logger)
            test_loader = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''
    model = define_Model(opt)
    model.init_train()
    logger.info(model.print_network())
    if opt['rank'] == 0 and opt['print']:
        logger.info(model.info_network())
        logger.info(model.info_params())
    print(model.netG)
    # return
    
    lpips_loss_fn = lpips.LPIPS(net='alex')
    if torch.cuda.is_available():
        lpips_loss_fn = lpips_loss_fn.cuda()
    
    ''' 
    # ----------------------------------------
    # Step--4 (main training)
    # ----------------------------------------
    '''
    best_psnr = 0.
    for epoch in range(1000000):  # keep running
        if opt['dist']:
            train_sampler.set_epoch(epoch + seed)
        for i, train_data in enumerate(train_loader):
            current_step += 1
            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)
            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters(current_step)

            # -------------------------------
            # 1) update learning rate
            # Adjust the learning rate update position here to after the optim update.
            # -------------------------------
            model.update_learning_rate(current_step)

            # -------------------------------
            # 4) training information
            # -------------------------------
            if current_step % opt['train']['checkpoint_print'] == 0 and opt['rank'] == 0:
                logs = model.current_log()  # such as loss
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(epoch, current_step, model.current_learning_rate())
                for k, v in logs.items():  # merge log information into message
                    message += '{:s}: {:.3e} '.format(k, v)
                logger.info(message)

            # -------------------------------
            # 5) save model
            # -------------------------------
            if current_step % opt['train']['checkpoint_save'] == 0 and opt['rank'] == 0:
                logger.info('Saving the model.')
                model.save(current_step, False)

            # -------------------------------
            # 6) testing
            # -------------------------------
            if current_step % opt['train']['checkpoint_test'] == 0 and opt['rank'] == 0:

                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_lpips = 0.0
                uiqm_list = []
                ssims, psnrs = [], []
                idx = 0
                for test_data in test_loader:
                    idx += 1
                    image_name_ext = os.path.basename(test_data['L_path'][0])
                    img_name, ext = os.path.splitext(image_name_ext)

                    img_dir = os.path.join(opt['path']['images'], img_name)
                    util.mkdir(img_dir)
                    
                    model.feed_data(test_data)
                    model.test()

                    visuals = model.current_visuals()
                    
                    # -----------------------
                    # calculate LPIPS
                    # -----------------------
                    E_tensor = visuals['E']
                    H_tensor = visuals['H']
                    if E_tensor.dim() == 3:
                        E_tensor = E_tensor.unsqueeze(0)
                        H_tensor = H_tensor.unsqueeze(0)
                    E_tensor = E_tensor * 2.0 - 1.0
                    H_tensor = H_tensor * 2.0 - 1.0
                    
                    if torch.cuda.is_available() and E_tensor.device != torch.device('cuda'):
                        E_tensor = E_tensor.cuda()
                        H_tensor = H_tensor.cuda()
                    
                    with torch.no_grad():
                        current_lpips = lpips_loss_fn(H_tensor, E_tensor).item()
                    
                    E_img = util.tensor2uint(visuals['E'])
                    H_img = util.tensor2uint(visuals['H'])

                    # -----------------------
                    # save estimated image E
                    # -----------------------
                    # save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    # util.imsave(E_img, save_img_path)

                    # -----------------------
                    # calculate PSNR,SSIM,UIQM
                    # -----------------------
                    current_psnr = util.calculate_psnr(E_img, H_img, border=border)
                    current_ssim = util.calculate_ssim(E_img, H_img, border=border)
                    
                    # E_img = util.tensor2uint(test_data['L'])
                    # E_img = cv2.resize(E_img, (640, 480), interpolation=cv2.INTER_CUBIC)
                    
                    # current_psnr = getPSNR(E_img, H_img)
                    # current_ssim = getSSIM(E_img, H_img)
                    current_uiqm = uiqm.getUIQM(E_img)
                    
                    avg_psnr += current_psnr
                    avg_ssim += current_ssim
                    avg_lpips += current_lpips
                    uiqm_list.append(current_uiqm)

                    logger.info('{:->4d}--> {:>10s} | PSNR: {:<6.4f}dB; SSIM: {:<6.4f}; LPIPS: {:<6.4f}; UIQM: {:<6.4f}'.format(idx, image_name_ext, current_psnr, current_ssim, current_lpips, current_uiqm))

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_lpips = avg_lpips / idx
                
                # -----------------------
                # calculate uiqm
                # -----------------------
                # paths = sorted(glob(os.path.join(img_dir, '*.*')))
                # uiqms = []
                # for path in paths:
                #     # im = Image.open(path)
                #     img = cv2.imread(path)
                #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                #     uiqms.append(uiqm.getUIQM(np.array(img)))
                # avg_uiqm = np.array(uiqms)

                avg_uiqm = np.array(uiqm_list)
                
                # testing log
                logger.info('<epoch:{0:3d}, iter:{1:8,d}, Average PSNR : {2:<6.4f}dB, SSIM: {3:<6.4f}, LPIPS: {4:<6.4f}'.format(epoch, current_step, avg_psnr, avg_ssim, avg_lpips))
                logger.info('<epoch:{0:3d}, iter:{1:8,d}, UIQM Mean : {2:<6.4f}, std: {3:<6.4f}\n'.format(epoch, current_step, np.mean(avg_uiqm), np.std(avg_uiqm)))

                if best_psnr < avg_psnr:
                    best_psnr = avg_psnr
                    logger.info('Saving the best model.')
                    model.save(current_step, True)
    # Just for sign end of training
    logger.info('='*50)
    logger.info('end of train')
    logger.info('='*50)

if __name__ == '__main__':
    main()
