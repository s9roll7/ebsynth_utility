import os

from modules.ui import plaintext_to_html

import cv2
import glob

from extensions.ebsynth_utility.stage1 import ebsynth_utility_stage1
from extensions.ebsynth_utility.stage2 import ebsynth_utility_stage2
from extensions.ebsynth_utility.stage5 import ebsynth_utility_stage5
from extensions.ebsynth_utility.stage7 import ebsynth_utility_stage7

def x_ceiling(value, step):
  return -(-value // step) * step

def dump_dict(string, d:dict):
    for key in d.keys():
        string += ( key + " : " + str(d[key]) + "\n")
    return string

class debug_string:
    txt = ""
    def print(self, comment):
        self.txt += comment + '\n'
    def to_string(self):
        return self.txt

def ebsynth_utility_process(stage_index: int, project_dir:str, original_movie_path:str, key_min_gap:int, key_max_gap:int, key_th:float, key_add_last_frame:bool, blend_rate:float, no_mask_mode:bool):
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
    
    frame_path = os.path.join(project_dir , "video_frame")
    frame_mask_path = os.path.join(project_dir, "video_mask")
    org_key_path = os.path.join(project_dir, "video_key")
    img2img_key_path = os.path.join(project_dir, "img2img_key")
    img2img_upscale_key_path = os.path.join(project_dir, "img2img_upscale_key")

    if no_mask_mode:
        frame_mask_path = ""

    project_args = [project_dir, original_movie_path, frame_path, frame_mask_path, org_key_path, img2img_key_path, img2img_upscale_key_path]


    if stage_index == 0:
        ebsynth_utility_stage1(dbg, project_args)
    elif stage_index == 1:
        ebsynth_utility_stage2(dbg, project_args, key_min_gap, key_max_gap, key_th, key_add_last_frame)
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
        dbg.print("4. I recommend to fill in the \"Width\" field with [" + str(img_width) + "]" )
        dbg.print("5. I recommend to fill in the \"Height\" field with [" + str(img_height) + "]" )
        dbg.print("6. I recommend to fill in the \"Denoising strength\" field with lower than 0.35" )
        dbg.print("   (It's okay to put a Large value in \"Face Denoising Strength\")")
        dbg.print("7. Fill in the remaining configuration fields of img2img. No image and mask settings are required.")
        dbg.print("8. Generate")
        dbg.print("(Images are output to [" + img2img_key_path + "])")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return process_end( dbg, "" )
    elif stage_index == 3:
        sample_image = glob.glob( os.path.join(frame_path , "*.png" ) )[0]
        img_height, img_width, _ = cv2.imread(sample_image).shape
        dbg.print("stage 4")
        dbg.print("")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        dbg.print("0. Enable the following item")
        dbg.print("Settings ->")
        dbg.print("  Saving images/grids ->")
        dbg.print("    Use original name for output filename during batch process in extras tab")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        dbg.print("1. Go to Extras tab")
        dbg.print("2. Go to Batch from Directory tab")
        dbg.print("3. Fill in the \"Input directory\" field with [" + img2img_key_path + "]" )
        dbg.print("4. Fill in the \"Output directory\" field with [" + img2img_upscale_key_path + "]" )
        dbg.print("5. Go to Scale to tab")
        dbg.print("6. Fill in the \"Width\" field with [" + str(img_width) + "]" )
        dbg.print("7. Fill in the \"Height\" field with [" + str(img_height) + "]" )
        dbg.print("8. Fill in the remaining configuration fields of Upscaler.")
        dbg.print("9. Generate")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return process_end( dbg, "" )
    elif stage_index == 4:
        ebsynth_utility_stage5(dbg, project_args)
    elif stage_index == 5:
        dbg.print("stage 6")
        dbg.print("")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        dbg.print("Running ebsynth.(on your self)")
        dbg.print("Open the generated .ebs under project directory and press [Run All] button.")
        dbg.print("If ""out-*"" directory already exists in the Project directory, delete it manually before executing.")
        dbg.print("If multiple .ebs files are generated, run them all.")
        dbg.print("(I recommend associating the .ebs file with EbSynth.exe.)")
        dbg.print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return process_end( dbg, "" )
    elif stage_index == 6:
        ebsynth_utility_stage7(dbg, project_args, blend_rate)
    else:
        pass

    return process_end( dbg, info )
