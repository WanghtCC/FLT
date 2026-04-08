import os
import cv2
import torch
import random
import imageio
import torch.utils.data as data
import utils.utils_image as util
from PIL import Image, ImageFilter, ImageOps

import numpy as np
import skimage.transform

class DatasetUFO120(data.Dataset):
    def __init__(self, opt, logger):
        super(DatasetUFO120, self).__init__()
        self.opt = opt
        self.logger = logger
        self.sf = opt['scale'] if opt['scale'] else 4
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        # scale = 2 (320, 240) => (640, 480)
        # scale = 3 (214, 160) => (640, 480)
        # scale = 4 (160, 120) => (640, 480)
        if self.sf == 2: self.lr_res_ = (240, 320)
        elif self.sf == 3: self.lr_res_ = (160, 214)
        elif self.sf == 4: self.lr_res_ = (120, 160)
        elif self.sf == 8: self.lr_res_ = (60, 80)
        self.hr_folder = 'hr/'
        self.phase = opt['phase']
        self.get_all_paths(opt['dataroot_H'])

    def get_all_paths(self, dataroot):
        self.num_train, self.num_val, self.num_test = 0, 0, 0
        self.train_hr_paths, self.val_hr_paths, self.test_hr_paths = [], [], []

        if self.phase == 'train' or self.phase == 'val':
            # train/val paths
            data_dir = os.path.join(dataroot, 'train_val/')
            hr_path = sorted(os.listdir(data_dir + self.hr_folder))
            num_paths = len(hr_path)
            all_idx = list(range(num_paths))
            # 95% train, 5% val
            random.shuffle(all_idx)
            self.num_train = int(num_paths * 1) # training set ratio
            self.num_val = num_paths - self.num_train
            train_idx = set(all_idx[:self.num_train])
            # split data paths to training and validation sets
            for i in range(num_paths):
                if i in train_idx:
                    self.train_hr_paths.append(data_dir + self.hr_folder + hr_path[i])
                else:
                    self.val_hr_paths.append(data_dir + self.hr_folder + hr_path[i])
            self.logger.info('Loaded {0} samples for training'.format(self.num_train))
            if self.num_val > 0:
                self.logger.info('Loaded {0} samples for validation'.format(self.num_val))
        elif self.phase == 'test':
            # test paths
            test_dir = os.path.join(dataroot, 'TEST/')
            if os.path.exists(test_dir) and os.path.isdir(test_dir):
                test_hr_path = sorted(os.listdir(test_dir + self.hr_folder))
                self.test_hr_paths = [test_dir + self.hr_folder + path for path in test_hr_path]
                self.num_test = len(self.test_hr_paths)
                self.logger.info('Loaded {0} samples for testing'.format(self.num_test))
            else:
                self.logger.warning('Test directory {} does not exist or is not a directory'.format(test_dir))


    def __len__(self):
        if self.phase == 'train':
            return self.num_train
        elif self.phase == 'val':
            return self.num_val
        else:  # test
            return self.num_test
        
    def __getitem__(self, idx):
        if self.phase == 'train':
            H_path = self.train_hr_paths[idx]
        elif self.phase == 'val':
            H_path = self.val_hr_paths[idx]
        elif self.phase == 'test':
            H_path = self.test_hr_paths[idx]
            
        hr_res = (self.lr_res_[0] * self.sf, self.lr_res_[1] * self.sf)
        
        img_L = self.read_and_resize(H_path, res=self.lr_res_)
        img_H = self.read_and_resize(H_path, res=hr_res)

        imgs_L = self.preprocess(np.array(img_L))
        imgs_H = self.preprocess(np.array(img_H))
        
        imgs_L = torch.from_numpy(imgs_L).permute(2, 0, 1).float()
        imgs_H = torch.from_numpy(imgs_H).permute(2, 0, 1).float()
        print(imgs_L.shape, imgs_H.shape)
        return {'L': imgs_L, 'H': imgs_H, 'L_path': H_path, 'H_path': H_path}
    
    def preprocess(self, x):
        # return (x/127.5) - 1.0
        return x/255.

    def read_and_resize(self, paths, res=(480, 640), mode_='RGB'):
        # img = imageio.imread(paths, pilmode=mode_).astype(np.float)
        # img = skimage.transform.resize(img, res)
        # print(paths)
        img = cv2.imread(paths)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (res[1], res[0]), interpolation=cv2.INTER_CUBIC)
        return img


class DatasetKLSG(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt, logger):
        super(DatasetKLSG, self).__init__()
        self.opt = opt
        self.logger = logger
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.L_size = self.patch_size // self.sf
        self.class_names = ['plane', 'seafloor', 'ship']
        self.class_num = len(self.class_names)
        self.augment_total = 1500
        self.train_num = 1400
        self.test_num = 100

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_H = None
        self.paths_L = None
        if os.path.isdir(opt['dataroot_H']):
            self.paths_H = util.get_image_paths(opt['dataroot_H'])
        else:
            if self.opt['phase'] == 'test': raise Exception('Error: testdir is not a directory')
            self.logger.error('dataroot: {} does not exist'.format(opt['dataroot_H']))
            self.__create__()
            self.paths_H = util.get_image_paths(self.opt['dataroot_H'])

        assert self.paths_H, 'Error: H path is empty.'

    def __getitem__(self, index):

        L_path = None
        # ------------------------------------
        # get H image
        # ------------------------------------
        H_path = self.paths_H[index]

        img_H = util.imread_uint(H_path, self.n_channels)
        img_H = util.uint2single(img_H)

        # ------------------------------------
        # modcrop
        # ------------------------------------
        img_H = util.modcrop(img_H, self.sf)

        # ------------------------------------
        # get L image
        # ------------------------------------
        if self.paths_L:
            # --------------------------------
            # directly load L image
            # --------------------------------
            L_path = self.paths_L[index]
            img_L = util.imread_uint(L_path, self.n_channels)
            img_L = util.uint2single(img_L)

        else:
            # --------------------------------
            # sythesize L image via matlab's bicubic
            # --------------------------------
            # H, W = img_H.shape[:2]
            img_L = util.imresize_np(img_H, 1 / self.sf, True)


        # ------------------------------------
        # if train, get L/H patch pair
        # ------------------------------------
        if self.opt['phase'] == 'train':

            H, W, C = img_L.shape

            # --------------------------------
            # randomly crop the L patch
            # --------------------------------
            rnd_h = random.randint(0, max(0, H - self.L_size))
            rnd_w = random.randint(0, max(0, W - self.L_size))
            img_L = img_L[rnd_h:rnd_h + self.L_size, rnd_w:rnd_w + self.L_size, :]

            # --------------------------------
            # crop corresponding H patch
            # --------------------------------
            rnd_h_H, rnd_w_H = int(rnd_h * self.sf), int(rnd_w * self.sf)
            img_H = img_H[rnd_h_H:rnd_h_H + self.patch_size, rnd_w_H:rnd_w_H + self.patch_size, :]

            # --------------------------------
            # augmentation - flip and/or rotate
            # --------------------------------
            mode = random.randint(0, 7)
            img_L, img_H = util.augment_img(img_L, mode=mode), util.augment_img(img_H, mode=mode)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_H, img_L = util.single2tensor3(img_H), util.single2tensor3(img_L)

        if L_path is None:
            L_path = H_path

        return {'L': img_L, 'H': img_H, 'L_path': L_path, 'H_path': H_path}

    def __len__(self):
        return len(self.paths_H)

    def __create__(self):
        self.random_seed = random.randint(0, 10000)
        self.logger.info(f'dataset seed: {self.random_seed}')
        parent_folder = os.path.join('data/datasets', self.opt['dataset'])
        augmented_folder = os.path.join(self.opt['dataroot_H'], 'augmented')
        # os.makedirs(augmented_folder, exist_ok=True)

        # ------------------- 1. Extract 100 samples -------------------
        test_original_pool = self._extract_original_test_pool(parent_folder)
        self.logger.info(f"The original test set pool (100 sheets) has been created.")

        # ------------------- 2. Split Set6/Set60/Set100 -------------------
        set_configs = {
            'Set6': 6,
            'Set60': 60,
            'Set100': 100
        }
        test_sets = self._split_test_sets(test_original_pool, set_configs)

        # ------------------- 3. Data expansion (training set) -------------------
        train_original_per_class = self._get_train_original_images(parent_folder, test_original_pool)
        augmented_train = self._augment_train_data(train_original_per_class, augmented_folder)

        # ------------------- 4. Save images -------------------
        self._save_train_set(augmented_train)
        self._save_test_sets(test_sets)

        save_dir = os.path.join(self.opt['dataroot_H'], 'train')
        self.opt['dataroot_H'] = save_dir

    def _extract_original_test_pool(self, parent_folder):
        """Extract 100 original images from dataset (category balance)"""
        test_pool = []
        for cls in self.class_names:
            cls_path = os.path.join(parent_folder, cls)
            original_images = [
                os.path.join(cls_path, fname)
                for fname in os.listdir(cls_path)
                if util.is_image_file(fname)
            ]
            random.seed(self.random_seed)
            random.shuffle(original_images)
            # At least 1 sheet is selected for each category and distributed proportionally.
            min_per_class = 1
            max_per_class = len(original_images)
            per_class = min(max_per_class, (self.test_num // self.class_num) + 1)
            test_pool.extend(original_images[:per_class])
        # Randomly truncate to 100 images (ensure total is correct)
        random.shuffle(test_pool)
        return test_pool[:self.test_num]

    def _split_test_sets(self, test_pool, set_configs):
        """from 100 original test images, split into Set6/Set60/Set100"""
        test_sets = {}
        for set_name, size in set_configs.items():
            total_images = test_pool.copy()
            if size > len(total_images):
                raise ValueError(f"{set_name} needs {size} images, but the original test set only has {len(total_images)} images")
            # Ensure each set contains all categories (prioritize selecting 1 image from each category, then randomly fill the rest)
            class_images = {cls: [] for cls in self.class_names}
            for img in total_images:
                cls = img.split(os.sep)[-2]
                class_images[cls].append(img)
            selected = []
            # choose first 1 image from each category
            for cls in self.class_names:
                img = random.choice(class_images[cls])
                selected.append(img)
                total_images.remove(img)
            # fill the remaining slots (allowing duplicate selection)
            remaining = size - self.class_num
            if remaining > 0:
                random.shuffle(total_images)
                selected.extend(total_images[:remaining])
            test_sets[set_name] = selected
        return test_sets

    def _get_train_original_images(self, parent_folder, test_pool):
        """obtain the original images that are not selected for the test set (for training set augmentation)"""
        train_original = {cls: [] for cls in self.class_names}
        for cls in self.class_names:
            cls_path = os.path.join(parent_folder, cls)
            original_images = [
                os.path.join(cls_path, fname)
                for fname in os.listdir(cls_path)
                if util.is_image_file(fname)
            ]
            test_img_names = {os.path.basename(img) for img in test_pool if cls in img}
            train_original[cls] = [img for img in original_images if os.path.basename(img) not in test_img_names]
        return train_original

    def _augment_train_data(self, train_original, augmented_folder):
        """multi-scale cropping and data augmentation of original training images"""
        base_crop_sizes = [224, 192, 160]
        augmented = []
        per_class_augment = self.augment_total // self.class_num
        for cls, imgs in train_original.items():
            cls_dir = os.path.join(augmented_folder, cls)
            os.makedirs(cls_dir, exist_ok=True)
            for img_path in imgs:
                try:
                    img = Image.open(img_path).convert('RGB')
                except Exception as e:
                    self.logger.warning(f"Skipping damaged image: {img_path}, Error: {str(e)}")
                    continue
                # Dynamically get the available crop sizes for the current image (not exceeding the image dimensions)
                valid_sizes = [s for s in base_crop_sizes if s < img.width and s < img.height]
                if not valid_sizes:
                    print(f"Image {img_path} is too small ({img.width}x{img.height}), cannot be cropped, saving directly")
                    img.save(os.path.join(cls_dir, os.path.basename(img_path)))
                    augmented.append(os.path.join(cls_dir, os.path.basename(img_path)))
                    continue
                for _ in range(per_class_augment // len(imgs) + 1):  # Loop until the required number is met

                    if random.random() > 0.5: img = ImageOps.mirror(img)  # Mirror
                    img = img.rotate(random.choice([0, 90, 180, 270]))  # Rotate
                    # Randomly select a valid crop size
                    crop_size = random.choice(valid_sizes)
                    left = random.randint(0, img.width - crop_size)
                    top = random.randint(0, img.height - crop_size)
                    cropped_img = img.crop((left, top, left + crop_size, top + crop_size))
                    # Apply Unsharp Masking (50% probability)
                    if random.random() > 0.5:
                        cropped_img = cropped_img.filter(ImageFilter.UnsharpMask())
                    # Save augmented image
                    aug_name = f"{cls}_aug_{len(augmented)}.jpg"
                    aug_path = os.path.join(cls_dir, aug_name)
                    cropped_img.save(aug_path)
                    augmented.append(aug_path)
                    if len(augmented) >= self.augment_total:
                        break
                if len(augmented) >= self.augment_total:
                    break
        return augmented[:self.augment_total]  # Ensure not exceeding the target number

    def _save_train_set(self, train_images):
        """Save the training set (uniform size)"""
        save_dir = os.path.join(self.opt['dataroot_H'], 'train')
        os.makedirs(save_dir, exist_ok=True)
        h_size = self.opt['H_size']
        for img_path in train_images:
            img = cv2.imread(img_path)
            resized = cv2.resize(img, (h_size, h_size), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(os.path.join(save_dir, os.path.basename(img_path)), resized)

    def _save_test_sets(self, test_sets):
        """Save Set6/Set60/Set100 (original size + uniform size)"""
        h_size = self.opt['H_size']
        for set_name, images in test_sets.items():
            save_dir = os.path.join(self.opt['dataroot_H'], set_name)
            os.makedirs(save_dir, exist_ok=True)
            original_dir = os.path.join(save_dir, 'original')
            resized_dir = os.path.join(save_dir, 'resized')
            os.makedirs(original_dir, exist_ok=True)
            os.makedirs(resized_dir, exist_ok=True)
            for img_path in images:
                fname = os.path.basename(img_path)
                img = cv2.imread(img_path)
                cv2.imwrite(os.path.join(original_dir, fname), img)
                resized = cv2.resize(img, (h_size, h_size), interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(resized_dir, fname), resized)
    
    

__factory = {
    'KLSG': DatasetKLSG,
    'UFO': DatasetUFO120,
}

def get_names():
    return __factory.keys()

def init_dataset(opt, logger, *args, **kwargs):
    if opt['name'] not in __factory.keys():
        logger.error('Unknown dataset: {}'.format(opt['name']))
        raise KeyError("Unknown dataset: {}".format(opt['name']))
    return __factory[opt['name']](opt, logger, *args, **kwargs)

if __name__ == '__main__':
    print(get_names())
