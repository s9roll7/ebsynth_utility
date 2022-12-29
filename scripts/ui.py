
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
                            with gr.Accordion(label="stage 2"):
                                key_min_gap = gr.Slider(minimum=0, maximum=500, step=1, label='Minimum keyframe gap', value=10)
                                key_max_gap = gr.Slider(minimum=0, maximum=1000, step=1, label='Maximum keyframe gap', value=300)
                                key_th = gr.Slider(minimum=5.0, maximum=100.0, step=0.1, label='Threshold of delta frame edge', value=27.0)

                            with gr.Accordion(label="stage 7"):
                                blend_rate = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Crossfade blend rate', value=0.5)

                            with gr.Accordion(label="etc"):
                                no_mask_mode = gr.Checkbox(label="No Mask Mode", value=False)

                    with gr.Column(variant='panel'):
                        with gr.Column(scale=1):
                            with gr.Group():
                                debug_info = gr.HTML(elem_id="ebs_info_area", value=".")

                            with gr.Column(scale=2):
                                stage_index = gr.Radio(label='Process Stage', choices=["stage 1","stage 2","stage 3","stage 4","stage 5","stage 6","stage 7"], value="stage 1", type="index")
                                gr.HTML(value="<p style='margin-bottom: 0.7em'>\
                                                The process of creating a video can be divided into the following stages.<br><br>\
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
                                                    Composite audio files extracted from the original video onto the concatenated video.<br>\
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

                    key_min_gap,
                    key_max_gap,
                    key_th,

                    blend_rate,

                    no_mask_mode,

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

