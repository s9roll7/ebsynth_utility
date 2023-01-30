import os
import subprocess
import glob
import cv2
import re

def remove_pngs_in_dir(path):
    if not os.path.isdir(path):
        return
    
    pngs = glob.glob( os.path.join(path, "*.png") )
    for png in pngs:
        os.remove(png)

def ebsynth_utility_stage1(dbg, project_args, st1_mask_threshold, tb_use_fast_mode, tb_use_jit, is_invert_mask):
    dbg.print("stage1")
    dbg.print("")

    _, original_movie_path, frame_path, frame_mask_path, _, _, _ = project_args

    if is_invert_mask:
        if os.path.isdir( frame_path ) and os.path.isdir( frame_mask_path ):
            dbg.print("Skip as it appears that the frame and normal masks have already been generated.")
            return

    remove_pngs_in_dir(frame_path)

    if frame_mask_path:
        remove_pngs_in_dir(frame_mask_path)

    os.makedirs(frame_path, exist_ok=True)

    if frame_mask_path:
        os.makedirs(frame_mask_path, exist_ok=True)

    png_path = os.path.join(frame_path , "%05d.png")
    # ffmpeg.exe -ss 00:00:00  -y -i %1 -qscale 0 -f image2 -c:v png "%05d.png"
    subprocess.call("ffmpeg.exe -ss 00:00:00  -y -i " + original_movie_path + " -qscale 0 -f image2 -c:v png " + png_path, shell=True)

    dbg.print("frame extracted")

    if frame_mask_path:
        fast_str = " --fast" if tb_use_fast_mode else ""
        jit_str = " --jit" if tb_use_jit else ""
        subprocess.call("venv\\Scripts\\transparent-background --source " + frame_path + " --dest " + frame_mask_path + " --type map" + fast_str + jit_str, shell=True)

        mask_imgs = glob.glob( os.path.join(frame_mask_path, "*.png") )
        
        for m in mask_imgs:
            img = cv2.imread(m)
            img[img < int( 255 * st1_mask_threshold )] = 0
            cv2.imwrite(m, img)

        p = re.compile(r'([0-9]+)_[a-z]*\.png')

        for mask in mask_imgs:
            base_name = os.path.basename(mask)
            m = p.fullmatch(base_name)
            if m:
                os.rename(mask, os.path.join(frame_mask_path, m.group(1) + ".png"))
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
