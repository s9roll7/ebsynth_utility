# ebsynth_utility

## Overview
#### AUTOMATIC1111 UI extension for creating videos using img2img and ebsynth.
#### This extension allows you to output edited videos using ebsynth.(AE is not required)

## Example
#### sample 1
<div><video controls src="https://user-images.githubusercontent.com/118420657/209951020-bee819b7-b8ef-48a9-8630-0e7c5c9a5d2f.mp4" muted="false"></video></div>

#### sample 2
<div><video controls src="https://user-images.githubusercontent.com/118420657/209951127-ff61671c-31aa-4e11-82f2-b3d9212c8748.mp4" muted="false"></video></div>

## Installation
- Install [ffmpeg](https://ffmpeg.org/) for your operating system
  (https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)
- Install [Ebsynth](https://ebsynth.com/)
- Use the Extensions tab of the webui to [Install from URL]

## Usage
- Go to [Ebsynth Utility] tab.
- Create an empty directory somewhere, and fill in the "Project directory" field.
- Place the video you want to edit from somewhere, and fill in the "Original Movie Path" field.
  Use short videos of a few seconds at first.
- Select stage 1 and Generate.
- Execute in order from stage 1 to 7.
  Progress during the process is not reflected in webui, so please check the console screen.
  If you see "completed." in webui, it is completed.

## Note
For reference, here's what I did when I edited a 1280x720 30fps 15sec video based on
#### Stage 1
There is nothing to configure.  
All frames of the video and mask images for all frames are generated.  
  
#### Stage 2
In the implementation of this extension, the keyframe interval is chosen to be shorter where there is a lot of motion and longer where there is little motion.  
First, generate one time with the default settings and go straight ahead without worrying about the result.  
  
#### Stage 3
Select one of the keyframes, throw it to img2img, and run [Interrogate DeepBooru].  
Delete unwanted words such as blur from the displayed prompt.  
Fill in the rest of the settings as you would normally do for image generation.  
  
Here is the settings I used.  
- Sampling method : DDIM  
- Sampling Steps : 50  
- Width : 960  
- Height : 512  
- CFG Scale : 20  
- Denoising strength : 0.35  
  
Here is the settings for extension.  
- use Face Crop img2img : True  
- Face Detection Method : YuNet  
- Max Crop Size : 1024  
- Face Denoising Strength : 0.35  
- Face Area Magnification : 1.5  
- Enable Face Prompt : False  
  
Trial and error in this process is the most time-consuming part.  
Monitor the destination folder and if you do not like results, interrupt and change the settings.  
[Denoising strength] and [Face Denoising Strength] settings when using Face Crop img2img will greatly affect the result.  
For more information on Face Crop img2img, check [here](https://github.com/s9roll7/face_crop_img2img)
  
If you have lots of memory to spare, increasing the width and height values while maintaining the aspect ratio may greatly improve results.  
  
#### Stage 4
Scale it up or down and process it to exactly the same size as the original video.  
This process should only need to be done once.  
  
- Width : 1280  
- Height : 720  
- Upscaler 1 : R-ESRGAN 4x+  
- Upscaler 2 : R-ESRGAN 4x+ Anime6B  
- Upscaler 2 visibility : 0.5  
- GFPGAN visibility : 1  
- CodeFormer visibility : 0  
- CodeFormer weight : 0  
  
#### Stage 5
There is nothing to configure.  
.ebs file will be generated.  
  
#### Stage 6
Run the .ebs file.  
I wouldn't change the settings, but you could adjust the .ebs settings.  

#### Stage 7
Finally, output the video.  
In my case, the entire process from 1 to 7 took about 30 minutes.  
  
- Crossfade blend rate : 1.0  
- Export type : mp4  

