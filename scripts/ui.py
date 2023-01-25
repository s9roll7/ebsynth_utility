
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
                        with gr.TabItem('project setting', id='ebs_project_setting'):
                            project_dir = gr.Textbox(label='Project directory', lines=1)
                            original_movie_path = gr.Textbox(label='Original Movie Path', lines=1)
                        with gr.TabItem('configuration', id='ebs_configuration'):
                            with gr.Accordion(label="stage 1"):
                                # https://pypi.org/project/transparent-background/
                                gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                        configuration for \
                                        <font color=\"blue\"><a href=\"https://pypi.org/project/transparent-background\">[transparent-background]</a></font>\
                                        </p>")
                                tb_use_fast_mode = gr.Checkbox(label="Use Fast Mode(It will be faster, but the quality of the mask will be lower.)", value=False)
                                tb_use_jit = gr.Checkbox(label="Use Jit", value=False)

                            with gr.Accordion(label="stage 2"):
                                key_min_gap = gr.Slider(minimum=0, maximum=500, step=1, label='Minimum keyframe gap', value=10)
                                key_max_gap = gr.Slider(minimum=0, maximum=1000, step=1, label='Maximum keyframe gap', value=300)
                                key_th = gr.Slider(minimum=0.0, maximum=100.0, step=0.1, label='Threshold of delta frame edge', value=8.5)
                                key_add_last_frame = gr.Checkbox(label="Add last frame to keyframes", value=True)

                            with gr.Accordion(label="stage 7"):
                                blend_rate = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Crossfade blend rate', value=1.0)
                                export_type = gr.Dropdown(choices=["mp4","webm","gif","rawvideo"], value="mp4" ,label="Export type")

                            with gr.Accordion(label="stage 8"):
                                bg_src = gr.Textbox(label='Background source(mp4 or directory containing images)', lines=1)
                                bg_type = gr.Dropdown(choices=["Fit video length","Loop"], value="Fit video length" ,label="Background type")
                                mask_blur_size = gr.Slider(minimum=0, maximum=150, step=1, label='Mask Blur Kernel Size', value=5)

                            with gr.Accordion(label="etc"):
                                mask_mode = gr.Dropdown(choices=["Normal","Invert","None"], value="Normal" ,label="Mask Mode")

                    with gr.Column(variant='panel'):
                        with gr.Column(scale=1):
                            with gr.Group():
                                debug_info = gr.HTML(elem_id="ebs_info_area", value=".")

                            with gr.Column(scale=2):
                                stage_index = gr.Radio(label='Process Stage', choices=["stage 1","stage 2","stage 3","stage 4","stage 5","stage 6","stage 7","stage 8"], value="stage 1", type="index")
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

                    tb_use_fast_mode,
                    tb_use_jit,

                    key_min_gap,
                    key_max_gap,
                    key_th,
                    key_add_last_frame,

                    blend_rate,
                    export_type,

                    bg_src,
                    bg_type,
                    mask_blur_size,

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

