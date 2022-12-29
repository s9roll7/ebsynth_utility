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

def ebsynth_utility_stage1(dbg, project_args):
    dbg.print("stage1")
    dbg.print("")

    _, original_movie_path, frame_path, frame_mask_path, _, _, _ = project_args

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
        subprocess.call("venv\\Scripts\\transparent-background --source " + frame_path + " --dest " + frame_mask_path + " --type map --fast", shell=True)

        mask_imgs = glob.glob( os.path.join(frame_mask_path, "*.png") )
        
        for m in mask_imgs:
            img = cv2.imread(m)
            img[img < 30] = 0
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


