import os
import re
import subprocess
import glob
import shutil
import time
import cv2
import numpy as np


def clamp(n, smallest, largest):
    return sorted([smallest, n, largest])[1]


def create_movie_from_frames( dir, start, end, number_of_digits, fps, output_path, export_type):
    def get_export_str(export_type):
        if export_type == "mp4":
            return " -vcodec libx264 -pix_fmt yuv420p "
        elif export_type == "webm":
#            return " -vcodec vp9 -crf 10 -b:v 0 "
            return " -crf 40 -b:v 0 -threads 4 "
        elif export_type == "gif":
            return " "
        elif export_type == "rawvideo":
            return " -vcodec rawvideo -pix_fmt bgr24 "

    vframes = end - start + 1
    path = os.path.join(dir , '%0' + str(number_of_digits) + 'd.png')
    
    # ffmpeg -r 10 -start_number n -i snapshot_%03d.png -vframes 50 example.gif
    subprocess.call("ffmpeg -framerate " + str(fps) + " -r " + str(fps) +
                        " -start_number " + str(start) +
                        " -i " + path + 
                        " -vframes " + str( vframes ) +
                        get_export_str(export_type) +
                        output_path, shell=True)


def search_out_dirs(proj_dir, blend_rate):
    ### create out_dirs
    p = re.compile(r'.*[\\\/]out\-([0-9]+)[\\\/]')

    number_of_digits = -1
    
    out_dirs=[]
    for d in glob.glob( os.path.join(proj_dir ,"out-*/"), recursive=False):
        m = p.fullmatch(d)
        if m:
            if number_of_digits == -1:
                number_of_digits = len(m.group(1))
            out_dirs.append({ 'keyframe':int(m.group(1)), 'path':d })
    
    out_dirs = sorted(out_dirs, key=lambda x: x['keyframe'], reverse=True)
    
    print(number_of_digits)
    
    prev_key = -1
    for out_d in out_dirs:
        out_d['next_keyframe'] = prev_key
        prev_key = out_d['keyframe']
    
    out_dirs = sorted(out_dirs, key=lambda x: x['keyframe'])
    
    
    ### search start/end frame
    prev_key = 0
    for out_d in out_dirs:
        imgs = sorted(glob.glob(  os.path.join( out_d['path'], '[0-9]'*number_of_digits + '.png') ))
        
        first_img = imgs[0]
        print(first_img)
        basename_without_ext = os.path.splitext(os.path.basename(first_img))[0]
        blend_timing = (prev_key - out_d['keyframe'])*blend_rate + out_d['keyframe']
        blend_timing = round(blend_timing)
        start_frame = max( blend_timing, int(basename_without_ext) )
        out_d['startframe'] = start_frame
        
        last_img = imgs[-1]
        print(last_img)
        basename_without_ext = os.path.splitext(os.path.basename(last_img))[0]
        end_frame = min( out_d['next_keyframe'], int(basename_without_ext) )
        if end_frame == -1:
            end_frame = int(basename_without_ext)
        out_d['endframe'] = end_frame
        prev_key = out_d['keyframe']
    
    return number_of_digits, out_dirs

def get_ext(export_type):
    if export_type in ("mp4","webm","gif"):
        return "." + export_type
    else:
        return ".avi"

def trying_to_add_audio(original_movie_path, no_snd_movie_path, output_path, tmp_dir ):
    if os.path.isfile(original_movie_path):
        sound_path = os.path.join(tmp_dir , 'sound.mp4')
        subprocess.call("ffmpeg -i " + original_movie_path + " -vn -acodec copy " + sound_path, shell=True)
        
        if os.path.isfile(sound_path):
            # ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 output.mp4

            subprocess.call("ffmpeg -i " + no_snd_movie_path + " -i " + sound_path + " -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 " + output_path, shell=True)
            return True
    
    return False

def ebsynth_utility_stage7(dbg, project_args, blend_rate,export_type,is_invert_mask):
    dbg.print("stage7")
    dbg.print("")

    project_dir, original_movie_path, _, _, _, _, _ = project_args

    fps = 30
    clip = cv2.VideoCapture(original_movie_path)
    if clip:
        fps = clip.get(cv2.CAP_PROP_FPS)
        clip.release()
    
    blend_rate = clamp(blend_rate, 0.0, 1.0)

    dbg.print("blend_rate: {}".format(blend_rate))
    dbg.print("export_type: {}".format(export_type))
    dbg.print("fps: {}".format(fps))
    
    if is_invert_mask:
        project_dir = os.path.join( project_dir , "inv")

    tmp_dir = os.path.join( project_dir , "crossfade_tmp")

    
    if os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)
    
    number_of_digits, out_dirs = search_out_dirs( project_dir, blend_rate )
    
    if number_of_digits == -1:
        dbg.print('no out dir')
        return
    
    ### create frame imgs
    
    start = out_dirs[0]['startframe']
    end = out_dirs[-1]['endframe']
    
    cur_clip = 0
    next_clip = cur_clip+1 if len(out_dirs) > cur_clip+1 else -1
    
    current_frame = 0
    
    print(str(start) + " -> " + str(end+1))
    
    black_img = np.zeros_like( cv2.imread( os.path.join(out_dirs[cur_clip]['path'], str(start).zfill(number_of_digits) + ".png") ) )
    
    for i in range(start, end+1):
        
        print(str(i) + " / " + str(end))

        if next_clip == -1:
            break
        
        if i in range( out_dirs[cur_clip]['startframe'], out_dirs[cur_clip]['endframe'] +1):
            pass
        elif i in range( out_dirs[next_clip]['startframe'], out_dirs[next_clip]['endframe'] +1):
            cur_clip = next_clip
            next_clip = cur_clip+1 if len(out_dirs) > cur_clip+1 else -1
            if next_clip == -1:
                break
        else:
            ### black
            # front ... none
            # back ... none
            cv2.imwrite( os.path.join(tmp_dir, filename) , black_img)
            current_frame = i
            continue
        
        filename = str(i).zfill(number_of_digits) + ".png"
        
        # front ... cur_clip
        # back ... next_clip or none
        
        if i in range( out_dirs[next_clip]['startframe'], out_dirs[next_clip]['endframe'] +1):
            # front ... cur_clip
            # back ... next_clip
            img_f = cv2.imread( os.path.join(out_dirs[cur_clip]['path'] , filename) )
            img_b = cv2.imread( os.path.join(out_dirs[next_clip]['path'] , filename) )
            
            back_rate = (i - out_dirs[next_clip]['startframe'])/ max( 1 , (out_dirs[cur_clip]['endframe'] - out_dirs[next_clip]['startframe']) )
            
            img = cv2.addWeighted(img_f, 1.0 - back_rate, img_b, back_rate, 0)
            
            cv2.imwrite( os.path.join(tmp_dir , filename) , img)
        else:
            # front ... cur_clip
            # back ... none
            filename = str(i).zfill(number_of_digits) + ".png"
            shutil.copy( os.path.join(out_dirs[cur_clip]['path'] , filename) , os.path.join(tmp_dir , filename) )
        
        current_frame = i
    
    
    start2 = current_frame+1
    
    print(str(start2) + " -> " + str(end+1))
    
    for i in range(start2, end+1):
        filename = str(i).zfill(number_of_digits) + ".png"
        shutil.copy( os.path.join(out_dirs[cur_clip]['path'] , filename) , os.path.join(tmp_dir , filename) )
    
    ### create movie
    movie_base_name = time.strftime("%Y%m%d-%H%M%S")
    if is_invert_mask:
        movie_base_name = "inv_" + movie_base_name
    
    nosnd_path = os.path.join(project_dir , movie_base_name + get_ext(export_type))
    
    start = out_dirs[0]['startframe']
    end = out_dirs[-1]['endframe']

    create_movie_from_frames( tmp_dir, start, end, number_of_digits, fps, nosnd_path, export_type)

    dbg.print("exported : " + nosnd_path)
    
    if export_type == "mp4":

        with_snd_path = os.path.join(project_dir , movie_base_name + '_with_snd.mp4')

        if trying_to_add_audio(original_movie_path, nosnd_path, with_snd_path, tmp_dir):
            dbg.print("exported : " + with_snd_path)
    
    dbg.print("")
    dbg.print("completed.")

