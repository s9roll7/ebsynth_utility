import modules.scripts as scripts
import gradio as gr
import os
import torch
import random
import time
import pprint
import shutil

from modules.processing import process_images,Processed
from modules.paths import models_path
from modules.textual_inversion import autocrop
import modules.images
from modules import shared,deepbooru,masking
import cv2
import copy
import numpy as np
from PIL import Image,ImageOps
import glob
import requests
import json
import re
from extensions.ebsynth_utility.calculator import CalcParser,ParseError

def get_my_dir():
    if os.path.isdir("extensions/ebsynth_utility"):
        return "extensions/ebsynth_utility"
    return scripts.basedir()

def x_ceiling(value, step):
    return -(-value // step) * step

def remove_pngs_in_dir(path):
    if not os.path.isdir(path):
        return
    pngs = glob.glob( os.path.join(path, "*.png") )
    for png in pngs:
        os.remove(png)

def resize_img(img, w, h):
    if img.shape[0] + img.shape[1] < h + w:
        interpolation = interpolation=cv2.INTER_CUBIC
    else:
        interpolation = interpolation=cv2.INTER_AREA

    return cv2.resize(img, (w, h), interpolation=interpolation)

def download_and_cache_models(dirname):
    download_url = 'https://github.com/zymk9/yolov5_anime/blob/8b50add22dbd8224904221be3173390f56046794/weights/yolov5s_anime.pt?raw=true'
    model_file_name = 'yolov5s_anime.pt'

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    cache_file = os.path.join(dirname, model_file_name)
    if not os.path.exists(cache_file):
        print(f"downloading face detection model from '{download_url}' to '{cache_file}'")
        response = requests.get(download_url)
        with open(cache_file, "wb") as f:
            f.write(response.content)

    if os.path.exists(cache_file):
        return cache_file
    return None

class Script(scripts.Script):
    anime_face_detector = None
    face_detector = None
    face_merge_mask_filename = "face_crop_img2img_mask.png"
    face_merge_mask_image = None
    prompts_dir = ""
    calc_parser = None
    is_invert_mask = False
    controlnet_weight = 0.5
    controlnet_weight_for_face = 0.5
    add_tag_replace_underscore = False


# The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        return "ebsynth utility"

# Determines when the script should be shown in the dropdown menu via the 
# returned value. As an example:
# is_img2img is True if the current tab is img2img, and False if it is txt2img.
# Thus, return is_img2img to only show the script on the img2img tab.

    def show(self, is_img2img):
        return is_img2img

# How the script's is displayed in the UI. See https://gradio.app/docs/#components
# for the different UI components you can use and how to create them.
# Most UI components can return a value, such as a boolean for a checkbox.
# The returned values are passed to the run method as parameters.

    def ui(self, is_img2img):
        with gr.Column(variant='panel'):
            with gr.Column():
                project_dir = gr.Textbox(label='Project directory', lines=1)
                generation_test = gr.Checkbox(False, label="Generation TEST!!(Ignore Project directory and use the image and mask specified in the main UI)")

            with gr.Accordion("Mask option"):
                mask_mode = gr.Dropdown(choices=["Normal","Invert","None","Don't Override"], value="Normal" ,label="Mask Mode(Override img2img Mask mode)")
                inpaint_area = gr.Dropdown(choices=["Whole picture","Only masked","Don't Override"], type = "index", value="Only masked" ,label="Inpaint Area(Override img2img Inpaint area)")
                use_depth = gr.Checkbox(True, label="Use Depth Map If exists in /video_key_depth")
                gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                        See \
                        <font color=\"blue\"><a href=\"https://github.com/thygate/stable-diffusion-webui-depthmap-script\">[here]</a></font> for depth map.\
                        </p>")

            with gr.Accordion("ControlNet option"):
                controlnet_weight = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.5, label="Control Net Weight")
                controlnet_weight_for_face = gr.Slider(minimum=0.0, maximum=2.0, step=0.01, value=0.5, label="Control Net Weight For Face")
                use_preprocess_img = gr.Checkbox(True, label="Use Preprocess image If exists in /controlnet_preprocess")
                gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                        Please enable the following settings to use controlnet from this script.<br>\
                        <font color=\"red\">\
                        Settings->ControlNet->Allow other script to control this extension\
                        </font>\
                        </p>")
            
            with gr.Accordion("Loopback option"):
                img2img_repeat_count = gr.Slider(minimum=1, maximum=30, step=1, value=1, label="Img2Img Repeat Count (Loop Back)")
                inc_seed = gr.Slider(minimum=0, maximum=9999999, step=1, value=1, label="Add N to seed when repeating ")

            with gr.Accordion("Auto Tagging option"):
                auto_tag_mode = gr.Dropdown(choices=["None","DeepDanbooru","CLIP"], value="None" ,label="Auto Tagging")
                add_tag_to_head = gr.Checkbox(False, label="Add additional prompts to the head")
                add_tag_replace_underscore = gr.Checkbox(False, label="Replace '_' with ' '(Does not affect the function to add tokens using add_token.txt.)")
                gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                        The results are stored in timestamp_prompts.txt.<br>\
                        If you want to use the same tagging results the next time you run img2img, rename the file to prompts.txt<br>\
                        Recommend enabling the following settings.<br>\
                        <font color=\"red\">\
                        Settings->Interrogate Option->Interrogate: include ranks of model tags matches in results\
                        </font>\
                        </p>")

            with gr.Accordion("Face Crop option"):
                is_facecrop = gr.Checkbox(False, label="use Face Crop img2img")

                with gr.Row():
                    face_detection_method = gr.Dropdown(choices=["YuNet","Yolov5_anime"], value="YuNet" ,label="Face Detection Method")
                    gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                            If loading of the Yolov5_anime model fails, check\
                            <font color=\"blue\"><a href=\"https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/2235\">[this]</a></font> solution.\
                            </p>")
                face_crop_resolution = gr.Slider(minimum=128, maximum=2048, step=1, value=512, label="Face Crop Resolution")
                max_crop_size = gr.Slider(minimum=0, maximum=2048, step=1, value=1024, label="Max Crop Size")
                face_denoising_strength = gr.Slider(minimum=0.00, maximum=1.00, step=0.01, value=0.5, label="Face Denoising Strength")
                face_area_magnification = gr.Slider(minimum=1.00, maximum=10.00, step=0.01, value=1.5, label="Face Area Magnification ")
                disable_facecrop_lpbk_last_time = gr.Checkbox(False, label="Disable at the last loopback time")
                
                with gr.Column():
                    enable_face_prompt = gr.Checkbox(False, label="Enable Face Prompt")
                    face_prompt = gr.Textbox(label="Face Prompt", show_label=False, lines=2,
                        placeholder="Prompt for Face",
                        value = "face close up,"
                    )

        return [project_dir, generation_test, mask_mode, inpaint_area, use_depth, img2img_repeat_count, inc_seed, auto_tag_mode, add_tag_to_head, add_tag_replace_underscore, is_facecrop, face_detection_method, face_crop_resolution, max_crop_size, face_denoising_strength, face_area_magnification, enable_face_prompt, face_prompt, controlnet_weight, controlnet_weight_for_face, disable_facecrop_lpbk_last_time,use_preprocess_img]


    def detect_face_from_img(self, img_array):
        if not self.face_detector:
            dnn_model_path = autocrop.download_and_cache_models(os.path.join(models_path, "opencv"))
            self.face_detector = cv2.FaceDetectorYN.create(dnn_model_path, "", (0, 0))
        
        self.face_detector.setInputSize((img_array.shape[1], img_array.shape[0]))
        _, result = self.face_detector.detect(img_array)
        return result

    def detect_anime_face_from_img(self, img_array):
        import sys

        if not self.anime_face_detector:
            if 'models' in sys.modules:
                del sys.modules['models']

            anime_model_path = download_and_cache_models(os.path.join(models_path, "yolov5_anime"))

            if not os.path.isfile(anime_model_path):
                print( "WARNING!! " + anime_model_path + " not found.")
                print( "use YuNet instead.")
                return self.detect_face_from_img(img_array)

            self.anime_face_detector = torch.hub.load('ultralytics/yolov5', 'custom', path=anime_model_path)

            # warmup
            test = np.zeros([512,512,3],dtype=np.uint8)
            _ = self.anime_face_detector(test)

        result = self.anime_face_detector(img_array)
        #models.common.Detections
        faces = []
        for x_c, y_c, w, h, _, _ in result.xywh[0].tolist():
            faces.append( [ x_c - w/2 , y_c - h/2, w, h ] )

        return faces

    def detect_face(self, img, mask, face_detection_method, max_crop_size):
        img_array = np.array(img)

        # image without alpha
        if img_array.shape[2] == 4:
            img_array = img_array[:,:,:3]

        if mask is not None:
            if self.is_invert_mask:
                mask = ImageOps.invert(mask)
            mask_array = np.array(mask)/255
            if mask_array.ndim == 2:
                mask_array = mask_array[:, :, np.newaxis]

            if mask_array.shape[2] == 4:
                mask_array = mask_array[:,:,:3]
            
            img_array = mask_array * img_array
            img_array = img_array.astype(np.uint8)

        if face_detection_method == "YuNet":
            faces = self.detect_face_from_img(img_array)
        elif face_detection_method == "Yolov5_anime":
            faces = self.detect_anime_face_from_img(img_array)
        else:
            faces = self.detect_face_from_img(img_array)
        
        if faces is None or len(faces) == 0:
            return []
        
        face_coords = []
        for face in faces:
            x = int(face[0])
            y = int(face[1])
            w = int(face[2])
            h = int(face[3])
            if max(w,h) > max_crop_size:
                print("ignore big face")
                continue
            if w == 0 or h == 0:
                print("ignore w,h = 0 face")
                continue

            face_coords.append( [ x/img_array.shape[1],y/img_array.shape[0],w/img_array.shape[1],h/img_array.shape[0]] )

        return face_coords

    def get_mask(self):
        def create_mask( output, x_rate, y_rate, k_size ):
            img = np.zeros((512, 512, 3))
            img = cv2.ellipse(img, ((256, 256), (int(512 * x_rate), int(512 * y_rate)), 0), (255, 255, 255), thickness=-1)
            img = cv2.GaussianBlur(img, (k_size, k_size), 0)
            cv2.imwrite(output, img)
        
        if self.face_merge_mask_image is None:
            mask_file_path = os.path.join( get_my_dir() , self.face_merge_mask_filename)
            if not os.path.isfile(mask_file_path):
                create_mask( mask_file_path, 0.9, 0.9, 91)

            m = cv2.imread( mask_file_path )[:,:,0]
            m = m[:, :, np.newaxis]
            self.face_merge_mask_image = m / 255

        return self.face_merge_mask_image

    def face_img_crop(self, img, face_coords,face_area_magnification):
        img_array = np.array(img)
        face_imgs =[]
        new_coords = []

        for face in face_coords:
            x = int(face[0] * img_array.shape[1])
            y = int(face[1] * img_array.shape[0])
            w = int(face[2] * img_array.shape[1])
            h = int(face[3] * img_array.shape[0])
            print([x,y,w,h])

            cx = x + int(w/2)
            cy = y + int(h/2)

            x = cx - int(w*face_area_magnification / 2)
            x = x if x > 0 else 0
            w = cx + int(w*face_area_magnification / 2) - x
            w = w if x+w < img.width else img.width - x

            y = cy - int(h*face_area_magnification / 2)
            y = y if y > 0 else 0
            h = cy + int(h*face_area_magnification / 2) - y
            h = h if y+h < img.height else img.height - y

            print([x,y,w,h])

            face_imgs.append( img_array[y: y+h, x: x+w] )
            new_coords.append( [x,y,w,h] )
        
        resized = []
        for face_img in face_imgs:
            if face_img.shape[1] < face_img.shape[0]:
                re_w = self.face_crop_resolution
                re_h = int(x_ceiling( (self.face_crop_resolution / face_img.shape[1]) * face_img.shape[0] , 64))
            else:
                re_w = int(x_ceiling( (self.face_crop_resolution / face_img.shape[0]) * face_img.shape[1] , 64))
                re_h = self.face_crop_resolution
            
            face_img = resize_img(face_img, re_w, re_h)
            resized.append( Image.fromarray(face_img))

        return resized, new_coords

    def face_crop_img2img(self, p, face_coords, face_denoising_strength, face_area_magnification, enable_face_prompt, face_prompt, controlnet_input_img, controlnet_input_face_imgs, preprocess_img_exist):

        def merge_face(img, face_img, face_coord, base_img_size, mask):
            x_rate = img.width / base_img_size[0]
            y_rate = img.height / base_img_size[1]

            img_array = np.array(img)
            x = int(face_coord[0] * x_rate)
            y = int(face_coord[1] * y_rate)
            w = int(face_coord[2] * x_rate)
            h = int(face_coord[3] * y_rate)

            face_array = np.array(face_img)
            face_array = resize_img(face_array, w, h)
            mask = resize_img(mask, w, h)
            if mask.ndim == 2:
                mask = mask[:, :, np.newaxis]
            
            bg = img_array[y: y+h, x: x+w]
            img_array[y: y+h, x: x+w] = mask * face_array + (1-mask)*bg

            return Image.fromarray(img_array)

        base_img = p.init_images[0]

        base_img_size = (base_img.width, base_img.height)

        if face_coords is None or len(face_coords) == 0:
            print("no face detected")
            return process_images(p)

        print(face_coords)
        face_imgs, new_coords = self.face_img_crop(base_img, face_coords, face_area_magnification)

        if not face_imgs:
            return process_images(p)

        face_p = copy.copy(p)

        ### img2img base img
        proc = self.process_images(p, controlnet_input_img, self.controlnet_weight, preprocess_img_exist)
        print(proc.seed)

        ### img2img for each face
        face_img2img_results = []

        for face, coord, controlnet_input_face in zip(face_imgs, new_coords, controlnet_input_face_imgs):
            # cv2.imwrite("scripts/face.png", np.array(face)[:, :, ::-1])
            face_p.init_images = [face]
            face_p.width = face.width
            face_p.height = face.height
            face_p.denoising_strength = face_denoising_strength
            
            if enable_face_prompt:
                face_p.prompt = face_prompt
            else:
                face_p.prompt = "close-up face ," + face_p.prompt

            if p.image_mask is not None:
                x,y,w,h = coord
                cropped_face_mask = Image.fromarray(np.array(p.image_mask)[y: y+h, x: x+w])
                face_p.image_mask = modules.images.resize_image(0, cropped_face_mask, face.width, face.height)
            
            face_proc = self.process_images(face_p, controlnet_input_face, self.controlnet_weight_for_face, preprocess_img_exist)
            print(face_proc.seed)

            face_img2img_results.append((face_proc.images[0], coord))
        
        ### merge faces
        bg = proc.images[0]
        mask = self.get_mask()

        for face_img, coord in face_img2img_results:
            bg = merge_face(bg, face_img, coord, base_img_size, mask)
        
        proc.images[0] = bg

        return proc

    def get_depth_map(self, mask, depth_path ,img_basename, is_invert_mask):
        depth_img_path = os.path.join( depth_path , img_basename )

        depth = None

        if os.path.isfile( depth_img_path ):
            depth = Image.open(depth_img_path)
        else:
            # try 00001-0000.png
            os.path.splitext(img_basename)[0]
            depth_img_path = os.path.join( depth_path , os.path.splitext(img_basename)[0] + "-0000.png" )
            if os.path.isfile( depth_img_path ):
                depth = Image.open(depth_img_path)
        
        if depth:
            if mask:
                mask_array = np.array(mask)
                depth_array = np.array(depth)

                if is_invert_mask == False:
                    depth_array[mask_array[:,:,0] == 0] = 0
                else:
                    depth_array[mask_array[:,:,0] != 0] = 0

                depth = Image.fromarray(depth_array)

                tmp_path = os.path.join( depth_path , "tmp" )
                os.makedirs(tmp_path, exist_ok=True)
                tmp_path = os.path.join( tmp_path , img_basename )
                depth_array = depth_array.astype(np.uint16)
                cv2.imwrite(tmp_path, depth_array)

            mask = depth
        
        return depth!=None, mask
    
### auto tagging
    debug_count = 0

    def get_masked_image(self, image, mask_image):

        if mask_image == None:
            return image.convert("RGB")
        
        mask = mask_image.convert('L')
        if self.is_invert_mask:
            mask = ImageOps.invert(mask)
        crop_region = masking.get_crop_region(np.array(mask), 0)
#        crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
#        x1, y1, x2, y2 = crop_region
        image = image.crop(crop_region).convert("RGB")
        mask = mask.crop(crop_region)

        base_img = Image.new("RGB", image.size, (255, 190, 200))

        image = Image.composite( image, base_img, mask )

#        image.save("scripts/get_masked_image_test_"+ str(self.debug_count) + ".png")
#        self.debug_count += 1

        return image
    
    def interrogate_deepdanbooru(self, imgs, masks):
        prompts_dict = {}
        cause_err = False

        try:
            deepbooru.model.start()

            for img,mask in zip(imgs,masks):
                key = os.path.basename(img)
                print(key + " interrogate deepdanbooru")

                image = Image.open(img)
                mask_image = Image.open(mask) if mask else None
                image = self.get_masked_image(image, mask_image)

                prompt = deepbooru.model.tag_multi(image)

                prompts_dict[key] = prompt
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            cause_err = True
        finally:
            deepbooru.model.stop()
            if cause_err:
                print("Exception occurred during auto-tagging(deepdanbooru)")
                return Processed()

        return prompts_dict


    def interrogate_clip(self, imgs, masks):
        from modules import devices, shared, lowvram, paths
        import importlib
        import models

        caption_list = []
        prompts_dict = {}
        cause_err = False

        try:
            if shared.cmd_opts.lowvram or shared.cmd_opts.medvram:
                lowvram.send_everything_to_cpu()
                devices.torch_gc()

            with paths.Prioritize("BLIP"):
                importlib.reload(models)
                shared.interrogator.load()

            for img,mask in zip(imgs,masks):
                key = os.path.basename(img)
                print(key + " generate caption")

                image = Image.open(img)
                mask_image = Image.open(mask) if mask else None
                image = self.get_masked_image(image, mask_image)

                caption = shared.interrogator.generate_caption(image)
                caption_list.append(caption)

            shared.interrogator.send_blip_to_ram()
            devices.torch_gc()

            for img,mask,caption in zip(imgs,masks,caption_list):
                key = os.path.basename(img)
                print(key + " interrogate clip")

                image = Image.open(img)
                mask_image = Image.open(mask) if mask else None
                image = self.get_masked_image(image, mask_image)

                clip_image = shared.interrogator.clip_preprocess(image).unsqueeze(0).type(shared.interrogator.dtype).to(devices.device_interrogate)

                res = ""

                with torch.no_grad(), devices.autocast():
                    image_features = shared.interrogator.clip_model.encode_image(clip_image).type(shared.interrogator.dtype)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    for name, topn, items in shared.interrogator.categories():
                        matches = shared.interrogator.rank(image_features, items, top_count=topn)
                        for match, score in matches:
                            if shared.opts.interrogate_return_ranks:
                                res += f", ({match}:{score/100:.3f})"
                            else:
                                res += ", " + match

                prompts_dict[key] = (caption + res)

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(e)
            cause_err = True
        finally:
            shared.interrogator.unload()
            if cause_err:
                print("Exception occurred during auto-tagging(blip/clip)")
                return Processed()
        
        return prompts_dict


    def remove_reserved_token(self, token_list):
        reserved_list = ["pink_background","simple_background","pink","pink_theme"]

        result_list = []

        head_token = token_list[0]

        if head_token[2] == "normal":
            head_token_str = head_token[0].replace('pink background', '')
            token_list[0] = (head_token_str, head_token[1], head_token[2])

        for token in token_list:
            if token[0] in reserved_list:
                continue
            result_list.append(token)

        return result_list

    def remove_blacklisted_token(self, token_list):
        black_list_path = os.path.join(self.prompts_dir, "blacklist.txt") 
        if not os.path.isfile(black_list_path):
            print(black_list_path + " not found.")
            return token_list

        with open(black_list_path) as f:
            black_list = [s.strip() for s in f.readlines()]

            result_list = []

            for token in token_list:
                if token[0] in black_list:
                    continue
                result_list.append(token)
            
            token_list = result_list

        return token_list

    def add_token(self, token_list):
        add_list_path = os.path.join(self.prompts_dir, "add_token.txt") 
        if not os.path.isfile(add_list_path):
            print(add_list_path + " not found.")

            if self.add_tag_replace_underscore:
                token_list = [ (x[0].replace("_"," "), x[1], x[2]) for x in token_list ]

            return token_list
        
        if not self.calc_parser:
            self.calc_parser = CalcParser()

        with open(add_list_path) as f:
            add_list = json.load(f)
            '''
            [
                {
                    "target":"test_token",
                    "min_score":0.8,
                    "token": ["lora_name_A", "0.5"],
                    "type":"lora"
                },
                {
                    "target":"test_token",
                    "min_score":0.5,
                    "token": ["bbbb", "score - 0.1"],
                    "type":"normal"
                },
                {
                    "target":"test_token2",
                    "min_score":0.8,
                    "token": ["hypernet_name_A", "score"],
                    "type":"hypernet"
                },
                {
                    "target":"test_token3",
                    "min_score":0.0,
                    "token": ["dddd", "score"],
                    "type":"normal"
                }
            ]
            '''
            result_list = []

            for token in token_list:
                for add_item in add_list:
                    if token[0] == add_item["target"]:
                        if token[1] > add_item["min_score"]:
                            # hit
                            formula = str(add_item["token"][1])
                            formula = formula.replace("score",str(token[1]))
                            print('Input: %s' % str(add_item["token"][1]))

                            try:
                                score = self.calc_parser.parse(formula)
                                score = round(score, 3)
                            except (ParseError, ZeroDivisionError) as e:
                                print('Input: %s' % str(add_item["token"][1]))
                                print('Error: %s' % e)
                                print("ignore this token")
                                continue

                            print("score = " + str(score))
                            result_list.append( ( add_item["token"][0], score, add_item["type"] ) )
            
            if self.add_tag_replace_underscore:
                token_list = [ (x[0].replace("_"," "), x[1], x[2]) for x in token_list ]

            token_list = token_list + result_list

        return token_list

    def create_prompts_dict(self, imgs, masks, auto_tag_mode):
        prompts_dict = {}

        if auto_tag_mode == "DeepDanbooru":
            raw_dict = self.interrogate_deepdanbooru(imgs, masks)
        elif auto_tag_mode == "CLIP":
            raw_dict = self.interrogate_clip(imgs, masks)
        
        repatter = re.compile(r'\((.+)\:([0-9\.]+)\)')

        for key, value_str in raw_dict.items():
            value_list = [x.strip() for x in value_str.split(',')]

            value = []
            for v in value_list:
                m = repatter.fullmatch(v)
                if m:
                    value.append((m.group(1), float(m.group(2)), "normal"))
                else:
                    value.append((v, 1, "no_score"))
            
#            print(value)
            value = self.remove_reserved_token(value)
#            print(value)
            value = self.remove_blacklisted_token(value)
#            print(value)
            value = self.add_token(value)
#            print(value)

            def create_token_str(x):
                print(x)
                if x[2] == "no_score":
                    return x[0]
                elif x[2] == "lora":
                    return "<lora:" + x[0] + ":" + str(x[1]) + ">"
                elif x[2] == "hypernet":
                    return "<hypernet:" + x[0] + ":" + str(x[1]) + ">"
                else:
                    return "(" + x[0] + ":" + str(x[1]) + ")"

            value_list = [create_token_str(x) for x in value]
            value = ",".join(value_list)

            prompts_dict[key] = value

        return prompts_dict

    def load_prompts_dict(self, imgs, default_token):
        prompts_path = os.path.join(self.prompts_dir, "prompts.txt") 
        if not os.path.isfile(prompts_path):
            print(prompts_path + " not found.")
            return {}
        
        prompts_dict = {}

        print(prompts_path + " found!!")
        print("skip auto tagging.")
        
        with open(prompts_path) as f:
            raw_dict = json.load(f)
            prev_value = default_token
            for img in imgs:
                key = os.path.basename(img)

                if key in raw_dict:
                    prompts_dict[key] = raw_dict[key]
                    prev_value = raw_dict[key]
                else:
                    prompts_dict[key] = prev_value

        return prompts_dict
    
    def process_images(self, p, input_img, controlnet_weight, input_img_is_preprocessed):
        p.control_net_input_image = input_img
        p.control_net_weight = controlnet_weight
        if input_img_is_preprocessed:
            p.control_net_module = "none"
        return process_images(p)

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
# Custom functions can be defined here, and additional libraries can be imported 
# to be used in processing. The return value should be a Processed object, which is
# what is returned by the process_images method.
    def run(self, p, project_dir, generation_test, mask_mode, inpaint_area, use_depth, img2img_repeat_count, inc_seed, auto_tag_mode, add_tag_to_head, add_tag_replace_underscore, is_facecrop, face_detection_method, face_crop_resolution, max_crop_size, face_denoising_strength, face_area_magnification, enable_face_prompt, face_prompt, controlnet_weight, controlnet_weight_for_face, disable_facecrop_lpbk_last_time, use_preprocess_img):
        args = locals()

        if generation_test:
            print("generation_test")
            test_proj_dir = os.path.join( get_my_dir() , "generation_test_proj")
            os.makedirs(test_proj_dir, exist_ok=True)
            test_video_key_path = os.path.join( test_proj_dir , "video_key")
            os.makedirs(test_video_key_path, exist_ok=True)
            test_video_mask_path = os.path.join( test_proj_dir , "video_mask")
            os.makedirs(test_video_mask_path, exist_ok=True)

            controlnet_input_path = os.path.join(test_proj_dir, "controlnet_input")
            if os.path.isdir(controlnet_input_path):
                shutil.rmtree(controlnet_input_path)

            remove_pngs_in_dir(test_video_key_path)
            remove_pngs_in_dir(test_video_mask_path)

            test_base_img = p.init_images[0]
            test_mask = p.image_mask

            if test_base_img:
                test_base_img.save( os.path.join( test_video_key_path , "00001.png") )
            if test_mask:
                test_mask.save( os.path.join( test_video_mask_path , "00001.png") )
            
            project_dir = test_proj_dir
        else:
            if not os.path.isdir(project_dir):
                print("project_dir not found")
                return Processed()
        
        self.controlnet_weight = controlnet_weight
        self.controlnet_weight_for_face = controlnet_weight_for_face

        self.add_tag_replace_underscore = add_tag_replace_underscore
        self.face_crop_resolution = face_crop_resolution
        
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        if mask_mode == "Normal":
            p.inpainting_mask_invert = 0
        elif mask_mode == "Invert":
            p.inpainting_mask_invert = 1
        
        if inpaint_area in (0,1):  #"Whole picture","Only masked"
            p.inpaint_full_res = inpaint_area

        is_invert_mask = False
        if mask_mode == "Invert":
            is_invert_mask = True

            inv_path = os.path.join(project_dir, "inv")
            if not os.path.isdir(inv_path):
                print("project_dir/inv not found")
                return Processed()
            
            org_key_path = os.path.join(inv_path, "video_key")
            img2img_key_path = os.path.join(inv_path, "img2img_key")
            depth_path = os.path.join(inv_path, "video_key_depth")

            preprocess_path = os.path.join(inv_path, "controlnet_preprocess")

            controlnet_input_path = os.path.join(inv_path, "controlnet_input")

            self.prompts_dir = inv_path
            self.is_invert_mask = True
        else:
            org_key_path = os.path.join(project_dir, "video_key")
            img2img_key_path = os.path.join(project_dir, "img2img_key")
            depth_path = os.path.join(project_dir, "video_key_depth")

            preprocess_path = os.path.join(project_dir, "controlnet_preprocess")

            controlnet_input_path = os.path.join(project_dir, "controlnet_input")

            self.prompts_dir = project_dir
            self.is_invert_mask = False

        frame_mask_path = os.path.join(project_dir, "video_mask")

        if not use_depth:
            depth_path = None

        if not os.path.isdir(org_key_path):
            print(org_key_path + " not found")
            print("Generate key frames first." if is_invert_mask == False else \
                    "Generate key frames first.(with [Ebsynth Utility] Tab -> [configuration] -> [etc]-> [Mask Mode] = Invert setting)")
            return Processed()

        if not os.path.isdir(controlnet_input_path):
            print(controlnet_input_path + " not found")
            print("copy {0} -> {1}".format(org_key_path,controlnet_input_path))

            os.makedirs(controlnet_input_path, exist_ok=True)

            imgs = glob.glob( os.path.join(org_key_path ,"*.png") )
            for img in imgs:
                img_basename = os.path.basename(img)
                shutil.copy( img , os.path.join(controlnet_input_path, img_basename) )

        remove_pngs_in_dir(img2img_key_path)
        os.makedirs(img2img_key_path, exist_ok=True)


        def get_mask_of_img(img):
            img_basename = os.path.basename(img)
            
            if mask_mode != "None":
                mask_path = os.path.join( frame_mask_path , img_basename )
                if os.path.isfile( mask_path ):
                    return mask_path
            return ""
        
        def get_pair_of_img(img, target_dir):
            img_basename = os.path.basename(img)
            
            pair_path = os.path.join( target_dir , img_basename )
            if os.path.isfile( pair_path ):
                return pair_path
            print("!!! pair of "+ img + " not in " + target_dir)
            return ""

        def get_controlnet_input_img(img):
            pair_img = get_pair_of_img(img, controlnet_input_path)
            if not pair_img:
                pair_img = get_pair_of_img(img, org_key_path)
            return pair_img
        
        imgs = glob.glob( os.path.join(org_key_path ,"*.png") )
        masks = [ get_mask_of_img(i) for i in imgs ]
        controlnet_input_imgs = [ get_controlnet_input_img(i) for i in imgs ]

        for mask in masks:
            m = cv2.imread(mask) if mask else None
            if m is not None:
                if m.max() == 0:
                    print("{0} blank mask found".format(mask))
                    if m.ndim == 2:
                        m[0,0] = 255
                    else:
                        m = m[:,:,:3]
                        m[0,0,0:3] = 255
                    cv2.imwrite(mask, m)

        ######################
        # face crop
        face_coords_dict={}
        for img,mask in zip(imgs,masks):
            face_detected = False
            if is_facecrop:
                image = Image.open(img)
                mask_image = Image.open(mask) if mask else None
                face_coords = self.detect_face(image, mask_image, face_detection_method, max_crop_size)
                if face_coords is None or len(face_coords) == 0:
                    print("no face detected")
                else:
                    print("face detected")
                    face_detected = True
            
            key = os.path.basename(img)
            face_coords_dict[key] = face_coords if face_detected else []

        with open( os.path.join( project_dir if is_invert_mask == False else inv_path,"faces.txt" ), "w") as f:
            f.write(json.dumps(face_coords_dict,indent=4))

        ######################
        # prompts
        prompts_dict = self.load_prompts_dict(imgs, p.prompt)

        if not prompts_dict:
            if auto_tag_mode != "None":
                prompts_dict = self.create_prompts_dict(imgs, masks, auto_tag_mode)

                for key, value in prompts_dict.items():
                    prompts_dict[key] = (value + "," + p.prompt) if add_tag_to_head else (p.prompt + "," + value)

            else:
                for img in imgs:
                    key = os.path.basename(img)
                    prompts_dict[key] = p.prompt
            
        with open( os.path.join( project_dir if is_invert_mask == False else inv_path, time.strftime("%Y%m%d-%H%M%S_") + "prompts.txt" ), "w") as f:
            f.write(json.dumps(prompts_dict,indent=4))


        ######################
        # img2img
        for img, mask, controlnet_input_img, face_coords, prompts in zip(imgs, masks, controlnet_input_imgs, face_coords_dict.values(), prompts_dict.values()):

            # Generation cancelled.
            if shared.state.interrupted:
                print("Generation cancelled.")
                break

            image = Image.open(img)
            mask_image = Image.open(mask) if mask else None

            img_basename = os.path.basename(img)
            
            _p = copy.copy(p)
            
            _p.init_images=[image]
            _p.image_mask = mask_image
            _p.prompt = prompts
            resized_mask = None

            repeat_count = img2img_repeat_count
            
            if mask_mode != "None" or use_depth:
                if use_depth:
                    depth_found, _p.image_mask = self.get_depth_map( mask_image, depth_path ,img_basename, is_invert_mask )
                    mask_image = _p.image_mask
                    if depth_found:
                        _p.inpainting_mask_invert = 0
            
            preprocess_img_exist = False
            controlnet_input_base_img = Image.open(controlnet_input_img) if controlnet_input_img else None

            if use_preprocess_img:
                preprocess_img = os.path.join(preprocess_path, img_basename)
                if os.path.isfile( preprocess_img ):
                    controlnet_input_base_img = Image.open(preprocess_img)
                    preprocess_img_exist = True

            if face_coords:
                controlnet_input_face_imgs, _ = self.face_img_crop(controlnet_input_base_img, face_coords, face_area_magnification)

            while repeat_count > 0:

                if disable_facecrop_lpbk_last_time:
                    if img2img_repeat_count > 1:
                        if repeat_count == 1:
                            face_coords = None

                if face_coords:
                    proc = self.face_crop_img2img(_p, face_coords, face_denoising_strength, face_area_magnification, enable_face_prompt, face_prompt, controlnet_input_base_img, controlnet_input_face_imgs, preprocess_img_exist)
                else:
                    proc = self.process_images(_p, controlnet_input_base_img, self.controlnet_weight, preprocess_img_exist)
                    print(proc.seed)
                
                repeat_count -= 1

                if repeat_count > 0:
                    _p.init_images=[proc.images[0]]

                    if mask_image is not None and resized_mask is None:
                        resized_mask = resize_img(np.array(mask_image) , proc.images[0].width, proc.images[0].height)
                        resized_mask = Image.fromarray(resized_mask)
                    _p.image_mask = resized_mask
                    _p.seed += inc_seed

            proc.images[0].save( os.path.join( img2img_key_path , img_basename ) )

        with open( os.path.join( project_dir if is_invert_mask == False else inv_path,"param.txt" ), "w") as f:
            f.write(pprint.pformat(proc.info))
        with open( os.path.join( project_dir if is_invert_mask == False else inv_path ,"args.txt" ), "w") as f:
            f.write(pprint.pformat(args))

        return proc
