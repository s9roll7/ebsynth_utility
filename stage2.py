import cv2
import os
import glob
import shutil
import numpy as np
import math

#---------------------------------
# Copied from PySceneDetect
def mean_pixel_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (np.sum(np.abs(left.astype(np.int32) - right.astype(np.int32))) / num_pixels)


def estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return size

_kernel = None

def _detect_edges(lum: np.ndarray) -> np.ndarray:
    global _kernel
    """Detect edges using the luma channel of a frame.
    Arguments:
        lum: 2D 8-bit image representing the luma channel of a frame.
    Returns:
        2D 8-bit image of the same size as the input, where pixels with values of 255
        represent edges, and all other pixels are 0.
    """
    # Initialize kernel.
    if _kernel is None:
        kernel_size = estimated_kernel_size(lum.shape[1], lum.shape[0])
        _kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Estimate levels for thresholding.
    sigma: float = 1.0 / 3.0
    median = np.median(lum)
    low = int(max(0, (1.0 - sigma) * median))
    high = int(min(255, (1.0 + sigma) * median))

    # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
    # This increases edge overlap leading to improved robustness against noise and slow
    # camera movement. Note that very large kernel sizes can negatively affect accuracy.
    edges = cv2.Canny(lum, low, high)
    return cv2.dilate(edges, _kernel)

#---------------------------------

def detect_edges(img_path, mask_path, is_invert_mask):
    im = cv2.imread(img_path)
    if mask_path:
        mask = cv2.imread(mask_path)[:,:,0]
        mask = mask[:, :, np.newaxis]
        im = im * ( (mask == 0) if is_invert_mask else (mask > 0) )
#        im = im * (mask/255)
#        im = im.astype(np.uint8)
#        cv2.imwrite( os.path.join( os.path.dirname(mask_path) , "tmp.png" ) , im)

    hue, sat, lum = cv2.split(cv2.cvtColor( im , cv2.COLOR_BGR2HSV))
    return _detect_edges(lum)

def get_mask_path_of_img(img_path, mask_dir):
    img_basename = os.path.basename(img_path)
    mask_path = os.path.join( mask_dir , img_basename )
    return mask_path if os.path.isfile( mask_path ) else None

def analyze_key_frames(png_dir, mask_dir, th, min_gap, max_gap, add_last_frame, is_invert_mask):
    keys = []
    
    frames = sorted(glob.glob( os.path.join(png_dir, "[0-9]*.png") ))
    
    key_frame = frames[0]
    keys.append( int(os.path.splitext(os.path.basename(key_frame))[0]) )
    key_edges = detect_edges( key_frame, get_mask_path_of_img( key_frame, mask_dir ), is_invert_mask )
    gap = 0
    
    for frame in frames:
        gap += 1
        if gap < min_gap:
            continue
        
        edges = detect_edges( frame, get_mask_path_of_img( frame, mask_dir ), is_invert_mask )
        
        delta = mean_pixel_distance( edges, key_edges )
        
        _th = th * (max_gap - gap)/max_gap
        
        if _th < delta:
            basename_without_ext = os.path.splitext(os.path.basename(frame))[0]
            keys.append( int(basename_without_ext) )
            key_frame = frame
            key_edges = edges
            gap = 0
    
    if add_last_frame:
        basename_without_ext = os.path.splitext(os.path.basename(frames[-1]))[0]
        last_frame = int(basename_without_ext)
        if not last_frame in keys:
            keys.append( last_frame )

    return keys

def remove_pngs_in_dir(path):
    if not os.path.isdir(path):
        return
    
    pngs = glob.glob( os.path.join(path, "*.png") )
    for png in pngs:
        os.remove(png)

def ebsynth_utility_stage2(dbg, project_args, key_min_gap, key_max_gap, key_th, key_add_last_frame, is_invert_mask):
    dbg.print("stage2")
    dbg.print("")

    _, original_movie_path, frame_path, frame_mask_path, org_key_path, _, _ = project_args

    remove_pngs_in_dir(org_key_path)
    os.makedirs(org_key_path, exist_ok=True)

    fps = 30
    clip = cv2.VideoCapture(original_movie_path)
    if clip:
        fps = clip.get(cv2.CAP_PROP_FPS)
        clip.release()

    if key_min_gap == -1:
        key_min_gap = int(10 * fps/30)
    else:
        key_min_gap = max(1, key_min_gap)
        key_min_gap = int(key_min_gap * fps/30)
        
    if key_max_gap == -1:
        key_max_gap = int(300 * fps/30)
    else:
        key_max_gap = max(10, key_max_gap)
        key_max_gap = int(key_max_gap * fps/30)
    
    key_min_gap,key_max_gap = (key_min_gap,key_max_gap) if key_min_gap < key_max_gap else (key_max_gap,key_min_gap)
    
    dbg.print("fps: {}".format(fps))
    dbg.print("key_min_gap: {}".format(key_min_gap))
    dbg.print("key_max_gap: {}".format(key_max_gap))
    dbg.print("key_th: {}".format(key_th))

    keys = analyze_key_frames(frame_path, frame_mask_path, key_th, key_min_gap, key_max_gap, key_add_last_frame, is_invert_mask)

    dbg.print("keys : " + str(keys))
    
    for k in keys:
        filename = str(k).zfill(5) + ".png"
        shutil.copy( os.path.join( frame_path , filename) , os.path.join(org_key_path, filename) )


    dbg.print("")
    dbg.print("Keyframes are output to [" + org_key_path + "]")
    dbg.print("")
    dbg.print("[Ebsynth Utility]->[configuration]->[stage 2]->[Threshold of delta frame edge]")
    dbg.print("The smaller this value, the narrower the keyframe spacing, and if set to 0, the keyframes will be equally spaced at the value of [Minimum keyframe gap].")
    dbg.print("")
    dbg.print("If you do not like the selection, you can modify it manually.")
    dbg.print("(Delete keyframe, or Add keyframe from ["+frame_path+"])")

    dbg.print("")
    dbg.print("completed.")

