import os

from modules.ui import plaintext_to_html

import cv2
import glob

from extensions.ebsynth_utility.stage1 import ebsynth_utility_stage1,ebsynth_utility_stage1_invert
from extensions.ebsynth_utility.stage2 import ebsynth_utility_stage2
from extensions.ebsynth_utility.stage5 import ebsynth_utility_stage5
from extensions.ebsynth_utility.stage7 import ebsynth_utility_stage7
from extensions.ebsynth_utility.stage8 import ebsynth_utility_stage8


def x_ceiling(value, step):
  return -(-value // step) * step

def dump_dict(string, d:dict):
    for key in d.keys():
        string += ( key + " : " + str(d[key]) + "\n")
    return string

class debug_string:
    txt = ""
    def print(self, comment):
        print(comment)
        self.txt += comment + '\n'
    def to_string(self):
        return self.txt

def ebsynth_utility_process(stage_index: int, project_dir:str, original_movie_path:str, tb_use_fast_mode:bool, tb_use_jit:bool, key_min_gap:int, key_max_gap:int, key_th:float, key_add_last_frame:bool, blend_rate:float, export_type:str, bg_src:str, bg_type:str, mask_blur_size:int, mask_mode:str):
    args = locals()
    info = ""
    info = dump_dict(info, args)
    dbg = debug_string()

    def process_end(dbg, info):
        return plaintext_to_html(dbg.to_string()), plaintext_to_html(info)


    if not os.path.isdir(project_dir):
        dbg.print("project_dir not found")
        return process_end( dbg, info )

    if not os.path.isfile(original_movie_path):
        dbg.print("original_movie_path not found")
        return process_end( dbg, info )
    
    is_invert_mask = False
    if mask_mode == "Invert":
        is_invert_mask = True

    frame_path = os.path.join(project_dir , "video_frame")
    frame_mask_path = os.path.join(project_dir, "video_mask")

    if is_invert_mask:
        inv_path = os.path.join(project_dir, "inv")
        os.makedirs(inv_path, exist_ok=True)

        org_key_path = os.path.join(inv_path, "video_key")
        img2img_key_path = os.path.join(inv_path, "img2img_key")
        img2img_upscale_key_path = os.path.join(inv_path, "img2img_upscale_key")
    else:
        org_key_path = os.path.join(project_dir, "video_key")
        img2img_key_path = os.path.join(project_dir, "img2img_key")
        img2img_upscale_key_path = os.path.join(project_dir, "img2img_upscale_key")

    if mask_mode == "None":
        frame_mask_path = ""
    

    project_args = [project_dir, original_movie_path, frame_path, frame_mask_path, org_key_path, img2img_key_path, img2img_upscale_key_path]


    if stage_index == 0:
        ebsynth_utility_stage1(dbg, project_args, tb_use_fast_mode, tb_use_jit, is_invert_mask)
        if is_invert_mask:
            inv_mask_path = os.path.join(inv_path, "inv_video_mask")
            ebsynth_utility_stage1_invert(dbg, frame_mask_path, inv_mask_path)

    elif stage_index == 1:
        ebsynth_utility_stage2(dbg, project_args, key_min_gap, key_max_gap, key_th, key_add_last_frame, is_invert_mask)
    elif stage_index == 2:

        sample_image = glob.glob( os.path.join(frame_path , "*.png" ) )[0]
        img_height, img_width, _ = cv2.imread(sample_image).shape
        if img_width < img_height:
            re_w = 512
            re_h = int(x_ceiling( (512 / img_width) * img_height , 64))
        else:
            re_w = int(x_ceiling( (512 / img_height) * img_width , 64))
            re_h = 512
        img_width = re_w
        img_height = re_h

        dbg.print("stage 3")
        dbg.print("")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        dbg.print("1. Go to img2img tab")
        dbg.print("2. Select [ebsynth utility] in the script combo box")
        dbg.print("3. Fill in the \"Project directory\" field with [" + project_dir + "]" )
        dbg.print("4. Select in the \"Mask Mode(Override img2img Mask mode)\" field with [" + ("Invert" if is_invert_mask else "Normal") + "]" )
        dbg.print("5. I recommend to fill in the \"Width\" field with [" + str(img_width) + "]" )
        dbg.print("6. I recommend to fill in the \"Height\" field with [" + str(img_height) + "]" )
        dbg.print("7. I recommend to fill in the \"Denoising strength\" field with lower than 0.35" )
        dbg.print("   (It's okay to put a Large value in \"Face Denoising Strength\")")
        dbg.print("8. Fill in the remaining configuration fields of img2img. No image and mask settings are required.")
        dbg.print("9. Drop any image onto the img2img main screen. This is necessary to avoid errors, but does not affect the results of img2img.")
        dbg.print("10. Generate")
        dbg.print("(Images are output to [" + img2img_key_path + "])")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return process_end( dbg, "" )
    elif stage_index == 3:
        sample_image = glob.glob( os.path.join(frame_path , "*.png" ) )[0]
        img_height, img_width, _ = cv2.imread(sample_image).shape

        if is_invert_mask:
            project_dir = inv_path

        dbg.print("stage 4")
        dbg.print("")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        dbg.print("0. Enable the following item")
        dbg.print("Settings ->")
        dbg.print("  Saving images/grids ->")
        dbg.print("    Use original name for output filename during batch process in extras tab")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        dbg.print("1. If \"img2img_upscale_key\" directory already exists in the %s, delete it manually before executing."%(project_dir))
        dbg.print("2. Go to Extras tab")
        dbg.print("3. Go to Batch from Directory tab")
        dbg.print("4. Fill in the \"Input directory\" field with [" + img2img_key_path + "]" )
        dbg.print("5. Fill in the \"Output directory\" field with [" + img2img_upscale_key_path + "]" )
        dbg.print("6. Go to Scale to tab")
        dbg.print("7. Fill in the \"Width\" field with [" + str(img_width) + "]" )
        dbg.print("8. Fill in the \"Height\" field with [" + str(img_height) + "]" )
        dbg.print("9. Fill in the remaining configuration fields of Upscaler.")
        dbg.print("10. Generate")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return process_end( dbg, "" )
    elif stage_index == 4:
        ebsynth_utility_stage5(dbg, project_args, is_invert_mask)
    elif stage_index == 5:

        if is_invert_mask:
            project_dir = inv_path

        dbg.print("stage 6")
        dbg.print("")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        dbg.print("Running ebsynth.(on your self)")
        dbg.print("Open the generated .ebs under %s and press [Run All] button."%(project_dir))
        dbg.print("If ""out-*"" directory already exists in the %s, delete it manually before executing."%(project_dir))
        dbg.print("If multiple .ebs files are generated, run them all.")
        dbg.print("(I recommend associating the .ebs file with EbSynth.exe.)")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return process_end( dbg, "" )
    elif stage_index == 6:
        ebsynth_utility_stage7(dbg, project_args, blend_rate, export_type, is_invert_mask)
    elif stage_index == 7:
        if mask_mode != "Normal":
            dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            dbg.print("Please reset [configuration]->[etc]->[Mask Mode] to Normal.")
            dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            return process_end( dbg, "" )
        ebsynth_utility_stage8(dbg, project_args, bg_src, bg_type, mask_blur_size, export_type)
    else:
        pass

    return process_end( dbg, info )
