
import gradio as gr

from ebsynth_utility import ebsynth_utility_process
from modules import script_callbacks
from modules.call_queue import wrap_gradio_gpu_call

def on_ui_tabs():

    with gr.Blocks(analytics_enabled=False) as ebs_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):

                with gr.Row():
                    with gr.Tabs(elem_id="ebs_settings"):
                        with gr.TabItem('project setting', elem_id='ebs_project_setting'):
                            project_dir = gr.Textbox(label='Project directory', lines=1)
                            original_movie_path = gr.Textbox(label='Original Movie Path', lines=1)

                            org_video = gr.Video(interactive=True, mirror_webcam=False)
                            def fn_upload_org_video(video):
                                return video
                            org_video.upload(fn_upload_org_video, org_video, original_movie_path)
                            gr.HTML(value="<p style='margin-bottom: 1.2em'>\
                                    If you have trouble entering the video path manually, you can also use drag and drop.For large videos, please enter the path manually. \
                                    </p>")

                        with gr.TabItem('configuration', elem_id='ebs_configuration'):
                            with gr.Tabs(elem_id="ebs_configuration_tab"):
                                with gr.TabItem(label="stage 1",elem_id='ebs_configuration_tab1'):
                                    with gr.Row():
                                        frame_width = gr.Number(value=-1, label="Frame Width", precision=0, interactive=True)
                                        frame_height = gr.Number(value=-1, label="Frame Height", precision=0, interactive=True)
                                    gr.HTML(value="<p style='margin-bottom: 1.2em'>\
                                            -1 means that it is calculated automatically. If both are -1, the size will be the same as the source size. \
                                            </p>")

                                    st1_masking_method_index = gr.Radio(label='Masking Method', choices=["transparent-background","clipseg","transparent-background AND clipseg"], value="transparent-background", type="index")

                                    with gr.Accordion(label="transparent-background options"):
                                        st1_mask_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Mask Threshold', value=0.0)

                                        # https://pypi.org/project/transparent-background/
                                        gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                                configuration for \
                                                <font color=\"blue\"><a href=\"https://pypi.org/project/transparent-background\">[transparent-background]</a></font>\
                                                </p>")
                                        tb_use_fast_mode = gr.Checkbox(label="Use Fast Mode(It will be faster, but the quality of the mask will be lower.)", value=False)
                                        tb_use_jit = gr.Checkbox(label="Use Jit", value=False)

                                    with gr.Accordion(label="clipseg options"):
                                        clipseg_mask_prompt = gr.Textbox(label='Mask Target (e.g., girl, cats)', lines=1)
                                        clipseg_exclude_prompt = gr.Textbox(label='Exclude Target (e.g., finger, book)', lines=1)
                                        clipseg_mask_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Mask Threshold', value=0.4)
                                        clipseg_mask_blur_size = gr.Slider(minimum=0, maximum=150, step=1, label='Mask Blur Kernel Size(MedianBlur)', value=11)
                                        clipseg_mask_blur_size2 = gr.Slider(minimum=0, maximum=150, step=1, label='Mask Blur Kernel Size(GaussianBlur)', value=11)

                                with gr.TabItem(label="stage 2", elem_id='ebs_configuration_tab2'):
                                    key_min_gap = gr.Slider(minimum=0, maximum=500, step=1, label='Minimum keyframe gap', value=10)
                                    key_max_gap = gr.Slider(minimum=0, maximum=1000, step=1, label='Maximum keyframe gap', value=300)
                                    key_th = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, label='Threshold of delta frame edge', value=8.5)
                                    key_add_last_frame = gr.Checkbox(label="Add last frame to keyframes", value=True)

                                with gr.TabItem(label="stage 3.5", elem_id='ebs_configuration_tab3_5'):
                                    gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                            <font color=\"blue\"><a href=\"https://github.com/hahnec/color-matcher\">[color-matcher]</a></font>\
                                            </p>")
                                    
                                    color_matcher_method = gr.Radio(label='Color Transfer Method', choices=['default', 'hm', 'reinhard', 'mvgd', 'mkl', 'hm-mvgd-hm', 'hm-mkl-hm'], value="hm-mkl-hm", type="value")
                                    color_matcher_ref_type = gr.Radio(label='Color Matcher Ref Image Type', choices=['original video frame', 'first frame of img2img result'], value="original video frame", type="index")
                                    gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                            <font color=\"red\">If an image is specified below, it will be used with highest priority.</font>\
                                            </p>")
                                    color_matcher_ref_image = gr.Image(label="Color Matcher Ref Image", source='upload', mirror_webcam=False, type='pil')
                                    st3_5_use_mask = gr.Checkbox(label="Apply mask to the result", value=True)
                                    st3_5_use_mask_ref = gr.Checkbox(label="Apply mask to the Ref Image", value=False)
                                    st3_5_use_mask_org = gr.Checkbox(label="Apply mask to original image", value=False)
                                    #st3_5_number_of_itr = gr.Slider(minimum=1, maximum=10, step=1, label='Number of iterations', value=1)

                                with gr.TabItem(label="stage 7", elem_id='ebs_configuration_tab7'):
                                    blend_rate = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Crossfade blend rate', value=1.0)
                                    export_type = gr.Dropdown(choices=["mp4","webm","gif","rawvideo"], value="mp4" ,label="Export type")

                                with gr.TabItem(label="stage 8", elem_id='ebs_configuration_tab8'):
                                    bg_src = gr.Textbox(label='Background source(mp4 or directory containing images)', lines=1)
                                    bg_type = gr.Dropdown(choices=["Fit video length","Loop"], value="Fit video length" ,label="Background type")
                                    mask_blur_size = gr.Slider(minimum=0, maximum=150, step=1, label='Mask Blur Kernel Size', value=5)
                                    mask_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Mask Threshold', value=0.0)
                                    #is_transparent = gr.Checkbox(label="Is Transparent", value=True, visible = False)
                                    fg_transparency = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Foreground Transparency', value=0.0)

                                with gr.TabItem(label="etc", elem_id='ebs_configuration_tab_etc'):
                                    mask_mode = gr.Dropdown(choices=["Normal","Invert","None"], value="Normal" ,label="Mask Mode")

                    with gr.Column(variant='panel'):
                        with gr.Column(scale=1):
                            with gr.Group():
                                debug_info = gr.HTML(elem_id="ebs_info_area", value=".")

                            with gr.Column(scale=2):
                                stage_index = gr.Radio(label='Process Stage', choices=["stage 1","stage 2","stage 3","stage 3.5","stage 4","stage 5","stage 6","stage 7","stage 8"], value="stage 1", type="index")
                                gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                                The process of creating a video can be divided into the following stages.<br>\
                                                (Stage 3, 4, and 6 only show a guide and do nothing actual processing.)<br><br>\
                                                <b>stage 1</b> <br>\
                                                    Extract frames from the original video. <br>\
                                                    Generate a mask image. <br><br>\
                                                <b>stage 2</b> <br>\
                                                    Select keyframes to be given to ebsynth.<br><br>\
                                                <b>stage 3</b> <br>\
                                                    img2img keyframes.<br><br>\
                                                <b>stage 3.5</b> <br>\
                                                    (this is optional. Perform color correction on the img2img results and expect flickering to decrease. Or, you can simply change the color tone from the generated result.)<br><br>\
                                                <b>stage 4</b> <br>\
                                                    and upscale to the size of the original video.<br><br>\
                                                <b>stage 5</b> <br>\
                                                    Rename keyframes.<br>\
                                                    Generate .ebs file.(ebsynth project file)<br><br>\
                                                <b>stage 6</b> <br>\
                                                    Running ebsynth.(on your self)<br>\
                                                    Open the generated .ebs under project directory and press [Run All] button. <br>\
                                                    If ""out-*"" directory already exists in the Project directory, delete it manually before executing.<br>\
                                                    If multiple .ebs files are generated, run them all.<br><br>\
                                                <b>stage 7</b> <br>\
                                                    Concatenate each frame while crossfading.<br>\
                                                    Composite audio files extracted from the original video onto the concatenated video.<br><br>\
                                                <b>stage 8</b> <br>\
                                                    This is an extra stage.<br>\
                                                    You can put any image or images or video you like in the background.<br>\
                                                    You can specify in this field -> [Ebsynth Utility]->[configuration]->[stage 8]->[Background source]<br>\
                                                    If you have already created a background video in Invert Mask Mode([Ebsynth Utility]->[configuration]->[etc]->[Mask Mode]),<br>\
                                                    You can specify \"path_to_project_dir/inv/crossfade_tmp\".<br>\
                                                </p>")
                            
                            with gr.Row():
                                generate_btn = gr.Button('Generate', elem_id="ebs_generate_btn", variant='primary')
                            
                            with gr.Group():
                                html_info = gr.HTML()


            ebs_args = dict(
                fn=wrap_gradio_gpu_call(ebsynth_utility_process),
                inputs=[
                    stage_index,

                    project_dir,
                    original_movie_path,

                    frame_width,
                    frame_height,
                    st1_masking_method_index,
                    st1_mask_threshold,
                    tb_use_fast_mode,
                    tb_use_jit,
                    clipseg_mask_prompt,
                    clipseg_exclude_prompt,
                    clipseg_mask_threshold,
                    clipseg_mask_blur_size,
                    clipseg_mask_blur_size2,

                    key_min_gap,
                    key_max_gap,
                    key_th,
                    key_add_last_frame,

                    color_matcher_method,
                    st3_5_use_mask,
                    st3_5_use_mask_ref,
                    st3_5_use_mask_org,
                    color_matcher_ref_type,
                    color_matcher_ref_image,

                    blend_rate,
                    export_type,

                    bg_src,
                    bg_type,
                    mask_blur_size,
                    mask_threshold,
                    fg_transparency,

                    mask_mode,

                ],
                outputs=[
                    debug_info,
                    html_info,
                ],
                show_progress=False,
            )
            generate_btn.click(**ebs_args)
           
    return (ebs_interface, "Ebsynth Utility", "ebs_interface"),



script_callbacks.on_ui_tabs(on_ui_tabs)

