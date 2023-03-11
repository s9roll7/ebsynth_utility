import os
import subprocess
import glob
import cv2
import re

from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import torch
import numpy as np


def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)

def resize_all_img(path, frame_width, frame_height):
    if not os.path.isdir(path):
        return
    
    pngs = glob.glob( os.path.join(path, "*.png") )
    img = cv2.imread(pngs[0])
    org_h,org_w = img.shape[0],img.shape[1]

    if frame_width == -1 and frame_height == -1:
        return
    elif frame_width == -1 and frame_height != -1:
        frame_width = int(frame_height * org_w / org_h)
    elif frame_width != -1 and frame_height == -1:
        frame_height = int(frame_width * org_h / org_w)
    else:
        pass
    print("({0},{1}) resize to ({2},{3})".format(org_w, org_h, frame_width, frame_height))

    for png in pngs:
        img = cv2.imread(png)
        img = resize_img(img, frame_width, frame_height)
        cv2.imwrite(png, img)

def remove_pngs_in_dir(path):
    if not os.path.isdir(path):
        return
    
    pngs = glob.glob( os.path.join(path, "*.png") )
    for png in pngs:
        os.remove(png)

def create_and_mask(mask_dir1, mask_dir2, output_dir):
    masks = glob.glob( os.path.join(mask_dir1, "*.png") )

    for mask1 in masks:
        base_name = os.path.basename(mask1)
        print("combine {0}".format(base_name))
        
        mask2 = os.path.join(mask_dir2, base_name)
        if not os.path.isfile(mask2):
            print("{0} not found!!! -> skip".format(mask2))
            continue

        img_1 = cv2.imread(mask1)
        img_2 = cv2.imread(mask2)
        img_1 = np.minimum(img_1,img_2)

        out_path = os.path.join(output_dir, base_name)
        cv2.imwrite(out_path, img_1)


def create_mask_clipseg(input_dir, output_dir, clipseg_mask_prompt, clipseg_exclude_prompt, clipseg_mask_threshold, mask_blur_size, mask_blur_size2):
    from modules import devices

    devices.torch_gc()

    device = devices.get_optimal_device_name()

    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to(device)

    imgs = glob.glob( os.path.join(input_dir, "*.png") )
    texts = [x.strip() for x in clipseg_mask_prompt.split(',')]
    exclude_texts = [x.strip() for x in clipseg_exclude_prompt.split(',')] if clipseg_exclude_prompt else None
    
    if exclude_texts:
        all_texts = texts + exclude_texts
    else:
        all_texts = texts


    for img_count,img in enumerate(imgs):
        image = Image.open(img)
        base_name = os.path.basename(img)

        inputs = processor(text=all_texts, images=[image] * len(all_texts), padding="max_length", return_tensors="pt")
        inputs = inputs.to(device)

        with torch.no_grad(), devices.autocast():
            outputs = model(**inputs)
        
        if len(all_texts) == 1:
            preds = outputs.logits.unsqueeze(0)
        else:
            preds = outputs.logits

        mask_img = None

        for i in range(len(all_texts)):
            x = torch.sigmoid(preds[i])
            x = x.to('cpu').detach().numpy()

#            x[x < clipseg_mask_threshold] = 0
            x = x > clipseg_mask_threshold

            if i < len(texts):
                if mask_img is None:
                    mask_img = x
                else:
                    mask_img = np.maximum(mask_img,x)
            else:
                mask_img[x > 0] = 0

        mask_img = mask_img*255
        mask_img = mask_img.astype(np.uint8)
        
        if mask_blur_size > 0:
            mask_blur_size = mask_blur_size//2 * 2 + 1
            mask_img = cv2.medianBlur(mask_img, mask_blur_size)

        if mask_blur_size2 > 0:
            mask_blur_size2 = mask_blur_size2//2 * 2 + 1
            mask_img = cv2.GaussianBlur(mask_img, (mask_blur_size2, mask_blur_size2), 0)

        mask_img = resize_img(mask_img, image.width, image.height)

        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2RGB)
        save_path = os.path.join(output_dir, base_name)
        cv2.imwrite(save_path, mask_img)

        print("{0} / {1}".format( img_count+1,len(imgs) ))
    
    devices.torch_gc()


def create_mask_transparent_background(input_dir, output_dir, tb_use_fast_mode, tb_use_jit, st1_mask_threshold):
    fast_str = " --fast" if tb_use_fast_mode else ""
    jit_str = " --jit" if tb_use_jit else ""
    bin_path = os.path.join("venv", "Scripts")
    bin_path = os.path.join(bin_path, "transparent-background")

    if os.path.isfile(bin_path) or os.path.isfile(bin_path + ".exe"):
        subprocess.call(bin_path + " --source " + input_dir + " --dest " + output_dir + " --type map" + fast_str + jit_str, shell=True)
    else:
        subprocess.call("transparent-background --source " + input_dir + " --dest " + output_dir + " --type map" + fast_str + jit_str, shell=True)

    mask_imgs = glob.glob( os.path.join(output_dir, "*.png") )
    
    for m in mask_imgs:
        img = cv2.imread(m)
        img[img < int( 255 * st1_mask_threshold )] = 0
        cv2.imwrite(m, img)

    p = re.compile(r'([0-9]+)_[a-z]*\.png')

    for mask in mask_imgs:
        base_name = os.path.basename(mask)
        m = p.fullmatch(base_name)
        if m:
            os.rename(mask, os.path.join(output_dir, m.group(1) + ".png"))

def ebsynth_utility_stage1(dbg, project_args, frame_width, frame_height, st1_masking_method_index, st1_mask_threshold, tb_use_fast_mode, tb_use_jit, clipseg_mask_prompt, clipseg_exclude_prompt, clipseg_mask_threshold, clipseg_mask_blur_size, clipseg_mask_blur_size2, is_invert_mask):
    dbg.print("stage1")
    dbg.print("")

    if st1_masking_method_index == 1 and (not clipseg_mask_prompt):
        dbg.print("Error: clipseg_mask_prompt is Empty")
        return

    project_dir, original_movie_path, frame_path, frame_mask_path, _, _, _ = project_args

    if is_invert_mask:
        if os.path.isdir( frame_path ) and os.path.isdir( frame_mask_path ):
            dbg.print("Skip as it appears that the frame and normal masks have already been generated.")
            return

    # remove_pngs_in_dir(frame_path)

    if frame_mask_path:
        remove_pngs_in_dir(frame_mask_path)
    
    if frame_mask_path:
        os.makedirs(frame_mask_path, exist_ok=True)

    if os.path.isdir( frame_path ):
        dbg.print("Skip frame extraction")
    else:
        os.makedirs(frame_path, exist_ok=True)

        png_path = os.path.join(frame_path , "%05d.png")
        # ffmpeg.exe -ss 00:00:00  -y -i %1 -qscale 0 -f image2 -c:v png "%05d.png"
        subprocess.call("ffmpeg -ss 00:00:00  -y -i " + original_movie_path + " -qscale 0 -f image2 -c:v png " + png_path, shell=True)

        dbg.print("frame extracted")

        frame_width = max(frame_width,-1)
        frame_height = max(frame_height,-1)

        if frame_width != -1 or frame_height != -1:
            resize_all_img(frame_path, frame_width, frame_height)

    if frame_mask_path:
        if st1_masking_method_index == 0:
            create_mask_transparent_background(frame_path, frame_mask_path, tb_use_fast_mode, tb_use_jit, st1_mask_threshold)
        elif st1_masking_method_index == 1:
            create_mask_clipseg(frame_path, frame_mask_path, clipseg_mask_prompt, clipseg_exclude_prompt, clipseg_mask_threshold, clipseg_mask_blur_size, clipseg_mask_blur_size2)
        elif st1_masking_method_index == 2:
            tb_tmp_path = os.path.join(project_dir , "tb_mask_tmp")
            if not os.path.isdir( tb_tmp_path ):
                os.makedirs(tb_tmp_path, exist_ok=True)
                create_mask_transparent_background(frame_path, tb_tmp_path, tb_use_fast_mode, tb_use_jit, st1_mask_threshold)
            create_mask_clipseg(frame_path, frame_mask_path, clipseg_mask_prompt, clipseg_exclude_prompt, clipseg_mask_threshold, clipseg_mask_blur_size, clipseg_mask_blur_size2)
            create_and_mask(tb_tmp_path,frame_mask_path,frame_mask_path)


        dbg.print("mask created")
    
    dbg.print("")
    dbg.print("completed.")


def ebsynth_utility_stage1_invert(dbg, frame_mask_path, inv_mask_path):
    dbg.print("stage 1 create_invert_mask")
    dbg.print("")

    if not os.path.isdir( frame_mask_path ):
        dbg.print( frame_mask_path + " not found")
        dbg.print("Normal masks must be generated previously.")
        dbg.print("Do stage 1 with [Ebsynth Utility] Tab -> [configuration] -> [etc]-> [Mask Mode] = Normal setting first")
        return

    os.makedirs(inv_mask_path, exist_ok=True)

    mask_imgs = glob.glob( os.path.join(frame_mask_path, "*.png") )
    
    for m in mask_imgs:
        img = cv2.imread(m)
        inv = cv2.bitwise_not(img)

        base_name = os.path.basename(m)
        cv2.imwrite(os.path.join(inv_mask_path,base_name), inv)

    dbg.print("")
    dbg.print("completed.")
