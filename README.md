# ebsynth_utility_lite
Fork was created to facilitate the creation of videos via img2img based on the original [ebsynth_utility](https://github.com/s9roll7/ebsynth_utility)
<br>
## TODO
- [x] Delete script for img2img
- [ ] Add configuration → stage 5
- [ ] Stage 0 — changing the video size, for example from 1080x1920 to 512x904
- [ ] Stage 2 — manually add **custom_gap**
- [ ] Change Stage 3 for create a grid (min 1x1 max 3x3)
- [ ] Change Stage 4 for disassemble the grid back
- [ ] Stage 0 — add Presets (with changes via .json)
- [ ] Stage 5 — automatisation with Ebsynth? (Is it possible?)
- [ ] Edit **Readme.md**

#### If you want to help, feel free to create the [PR](https://github.com/alexbofa/ebsynth_utility_lite/pulls)

## Installation
- Install [ffmpeg](https://ffmpeg.org/) for your operating system
  (https://www.geeksforgeeks.org/how-to-install-ffmpeg-on-windows/)
- Install [Ebsynth](https://ebsynth.com/)
- Use the Extensions tab of the webui to [Install from URL]

## Usage
- Go to [Ebsynth Utility] tab.
- Create an empty directory somewhere, and fill in the «Project directory» field.
- Place the video you want to edit from somewhere, and fill in the "Original Movie Path" field.
  Use short videos of a few seconds at first.
- Select stage 1 and Generate.
- Execute in order from stage 1 to 7.
  Progress during the process is not reflected in webui, so please check the console screen.
  If you see "completed." in webui, it is completed.  
(In the current latest webui, it seems to cause an error if you do not drop the image on the main screen of img2img.  
Please drop the image as it does not affect the result.)  

<br>

## Note 1
For reference, here's what I did when I edited a 1280x720 30fps 15sec video based on
#### Stage 1
There is nothing to configure.  
All frames of the video and mask images for all frames are generated.  
  
#### Stage 2
In the implementation of this extension, the keyframe interval is chosen to be shorter where there is a lot of motion and longer where there is little motion.  
If the animation breaks up, increase the keyframe, if it flickers, decrease the keyframe.  
First, generate one time with the default settings and go straight ahead without worrying about the result.  

#### Stage 3 (In development)

#### Stage 4 (Will be changed in the future)
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
