import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

def init_metrics_model():
    lpips_model = lpips.LPIPS(net='alex')
    return lpips_model

def read_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Unable to read the image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, img_rgb.shape

def calc_single_pair_metrics(e_path, h_path, lpips_model):
    e_np, e_shape = read_image(e_path)
    h_np, h_shape = read_image(h_path)    
    if e_shape != h_shape:
        raise ValueError(f"Image dimensions do not match: {os.path.basename(e_path)} ({e_shape}) vs {os.path.basename(h_path)} ({h_shape})")
    # 1. Calculate PSNR   
    psnr_val = psnr(h_np, e_np, data_range=255)
    
    # 2. Calculate SSIM
    ssim_val = ssim(h_np, e_np, channel_axis=-1, data_range=255, multichannel=True)
    
    # 3. Calculate LPIPS
    e_tensor = lpips.im2tensor(e_np / 5).float()
    h_tensor = lpips.im2tensor(h_np / 5).float()
    lpips_val = lpips_model(e_tensor, h_tensor).item()

    return round(psnr_val, 2), round(ssim_val, 4), round(lpips_val, 4)

def main():
    # 1. Configure image path
    e_folder = 'log/sr-flt-ufo/flt_sr_x2/test_images/E'
    h_folder = 'log/sr-flt-ufo/flt_sr_x2/test_images/H'
    
    # 2. Initialize LPIPS model
    lpips_model = init_metrics_model()
    metrics_list = [] # [class, e_path, h_path, psnr, ssim, lpips]
    
    # 3. Batch calculation single-image indicators
    e_filenames = [f for f in os.listdir(e_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    h_filenames = [f for f in os.listdir(h_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    common_filenames = set(e_filenames) & set(h_filenames)
    if not common_filenames:
        raise ValueError("There are no matching image files in the two folders (check the file name and format)")
    if len(common_filenames) != len(e_filenames) or len(common_filenames) != len(h_filenames):
        print(f"Warning: Some files are not matched, and only the common documents are calculated. A total of {len(common_filenames)} pairs")
    
    for filename in sorted(common_filenames):
        e_path = os.path.join(e_folder, filename)
        h_path = os.path.join(h_folder, filename)
        
        category = filename.split('-')[0]
        
        try:
            psnr_val, ssim_val, lpips_val = calc_single_pair_metrics(e_path, h_path, lpips_model)
            metrics_list.append([category, filename, filename, psnr_val, ssim_val, lpips_val])
            print(f"{filename} | PSNR: {psnr_val} | SSIM: {ssim_val} | LPIPS: {lpips_val}")
        except Exception as e:
            print(f"{filename} | Error：{str(e)}")
            
    df = pd.DataFrame(metrics_list, columns=["class", "E_path", "H_path", "PSNR", "SSIM", "LPIPS"])
    
    # Calculate the average
    category_avg = df.groupby("class")[["PSNR", "SSIM", "LPIPS"]].mean().round(4)
    category_avg["samples"] = df.groupby("class").size().values
    
    # Calculate the total average
    total_avg = df[["PSNR", "SSIM", "LPIPS"]].mean().round(4)
    total_count = len(df)  # Total number of samples
    # Convert to DataFrame for easy merging and display
    total_avg = pd.DataFrame([total_avg], index=["Total Average"])
    total_avg["Sample Count"] = total_count
    
    print("Average Metrics by Category:")
    print(category_avg)
    print("Total Average:")
    print(total_avg)
                                                                 

if __name__ == "__main__":
    main()