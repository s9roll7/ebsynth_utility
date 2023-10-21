import cv2
import os
import glob
import shutil
import numpy as np
from PIL import Image

from color_matcher import ColorMatcher
from color_matcher.normalizer import Normalizer

def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)

def get_pair_of_img(img_path, target_dir):
    img_basename = os.path.basename(img_path)
    target_path = os.path.join( target_dir , img_basename )
    return target_path if os.path.isfile( target_path ) else None

def remove_pngs_in_dir(path):
    if not os.path.isdir(path):
        return
    
    pngs = glob.glob( os.path.join(path, "*.png") )
    for png in pngs:
        os.remove(png)

def get_pair_of_img(img, target_dir):
    img_basename = os.path.basename(img)
    
    pair_path = os.path.join( target_dir , img_basename )
    if os.path.isfile( pair_path ):
        return pair_path
    print("!!! pair of "+ img + " not in " + target_dir)
    return ""

def get_mask_array(mask_path):
    if not mask_path:
        return None
    mask_array = np.asarray(Image.open( mask_path ))
    if mask_array.ndim == 2:
        mask_array = mask_array[:, :, np.newaxis]
    mask_array = mask_array[:,:,:1]
    mask_array = mask_array/255
    return mask_array

def color_match(imgs, ref_image, color_matcher_method, dst_path):
    cm = ColorMatcher(method=color_matcher_method)

    i = 0
    total = len(imgs)

    for fname in imgs:

        img_src = Image.open(fname)
        img_src = Normalizer(np.asarray(img_src)).type_norm()

        img_src = cm.transfer(src=img_src, ref=ref_image, method=color_matcher_method)

        img_src = Normalizer(img_src).uint8_norm()
        Image.fromarray(img_src).save(os.path.join(dst_path, os.path.basename(fname)))

        i += 1
        print("{0}/{1}".format(i, total))

    imgs = sorted( glob.glob( os.path.join(dst_path, "*.png") ) )


def ebsynth_utility_stage3_5(dbg, project_args, color_matcher_method, st3_5_use_mask, st3_5_use_mask_ref, st3_5_use_mask_org, color_matcher_ref_type, color_matcher_ref_image):
    dbg.print("stage3.5")
    dbg.print("")

    _, _, frame_path, frame_mask_path, org_key_path, img2img_key_path, _ = project_args

    backup_path = os.path.join( os.path.join( img2img_key_path, "..") , "st3_5_backup_img2img_key")
    backup_path = os.path.normpath(backup_path)
    
    if not os.path.isdir( backup_path ):
        dbg.print("{0} not found -> create backup.".format(backup_path))
        os.makedirs(backup_path, exist_ok=True)

        imgs = glob.glob( os.path.join(img2img_key_path, "*.png") )

        for img in imgs:
            img_basename = os.path.basename(img)
            pair_path = os.path.join( backup_path , img_basename )            
            shutil.copy( img , pair_path)
    
    else:
        dbg.print("{0} found -> Treat the images here as originals.".format(backup_path))
    
    org_imgs = sorted( glob.glob( os.path.join(backup_path, "*.png") ) )
    head_of_keyframe = org_imgs[0]

    # open ref img
    ref_image = color_matcher_ref_image
    if not ref_image:
        dbg.print("color_matcher_ref_image not set")

        if color_matcher_ref_type == 0:
            #'original video frame'
            dbg.print("select -> original video frame")
            ref_image = Image.open( get_pair_of_img(head_of_keyframe, frame_path) )
        else:
            #'first frame of img2img result'
            dbg.print("select -> first frame of img2img result")
            ref_image = Image.open( get_pair_of_img(head_of_keyframe, backup_path) )

        ref_image = np.asarray(ref_image)

        if st3_5_use_mask_ref:
            mask = get_pair_of_img(head_of_keyframe, frame_mask_path)
            if mask:
                mask_array = get_mask_array( mask )
                ref_image = ref_image * mask_array
                ref_image = ref_image.astype(np.uint8)

    else:
        dbg.print("select -> color_matcher_ref_image")
        ref_image = np.asarray(ref_image)
    

    if color_matcher_method in ('mvgd', 'hm-mvgd-hm'):
        sample_img = Image.open(head_of_keyframe)
        ref_image = resize_img( ref_image, sample_img.width, sample_img.height )
    
    ref_image = Normalizer(ref_image).type_norm()
    

    if st3_5_use_mask_org:
        tmp_path = os.path.join( os.path.join( img2img_key_path, "..") , "st3_5_tmp")
        tmp_path = os.path.normpath(tmp_path)
        dbg.print("create {0} for masked original image".format(tmp_path))

        remove_pngs_in_dir(tmp_path)
        os.makedirs(tmp_path, exist_ok=True)

        for org_img in org_imgs:
            image_basename = os.path.basename(org_img)

            org_image = np.asarray(Image.open(org_img))

            mask = get_pair_of_img(org_img, frame_mask_path)
            if mask:
                mask_array = get_mask_array( mask )
                org_image = org_image * mask_array
                org_image = org_image.astype(np.uint8)

            Image.fromarray(org_image).save( os.path.join( tmp_path, image_basename ) )
        
        org_imgs = sorted( glob.glob( os.path.join(tmp_path, "*.png") ) )
    

    color_match(org_imgs, ref_image, color_matcher_method, img2img_key_path)


    if st3_5_use_mask or st3_5_use_mask_org:
        imgs = sorted( glob.glob( os.path.join(img2img_key_path, "*.png") ) )
        for img in imgs:
            mask = get_pair_of_img(img, frame_mask_path)
            if mask:
                mask_array = get_mask_array( mask )
                bg = get_pair_of_img(img, frame_path)
                bg_image = np.asarray(Image.open( bg ))
                fg_image = np.asarray(Image.open( img ))

                final_img = fg_image * mask_array + bg_image * (1-mask_array)
                final_img = final_img.astype(np.uint8)

                Image.fromarray(final_img).save(img)

    dbg.print("")
    dbg.print("completed.")

