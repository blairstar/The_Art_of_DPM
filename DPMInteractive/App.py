import random

import gradio as gr
import matplotlib.pyplot as plt

from DPMInteractive import g_st, g_et, g_num, g_res
from DPMInteractive import init_change, shrink_change, conv_change 
from DPMInteractive import cond_prob_init_change, cond_prob_alpha_change, cond_prob_cond_change
from DPMInteractive import forward_init_change, forward_seq_apply
from DPMInteractive import backward_seq_apply, fit_and_backward_apply
from DPMInteractive import contraction_init_change, contraction_alpha_change, change_two_inputs_seed, contraction_apply
from DPMInteractive import fixed_point_init_change, fixed_point_apply_iterate
from DPMInteractive import forward_plot_part, backward_plot_part, fit_plot_part, fixed_plot_part
from RenderMarkdown import md_introduction_block, md_transform_block, md_likelihood_block, md_posterior_block
from RenderMarkdown import md_forward_process_block, md_backward_process_block, md_fit_posterior_block
from RenderMarkdown import md_posterior_transform_block, md_deconvolution_block, md_cond_kl_block, md_approx_gauss_block
from RenderMarkdown import md_non_expanding_block, md_stationary_block, md_reference_block, md_about_block
from Misc import g_css, js_head, js_load


def gr_empty_space(size=1):
    space = gr.Markdown(" "*size, elem_classes="bgc")
    return space


def gr_number(label=None, minimum=None, maximum=None, value=None, step=1.0, precision=0, min_width=160):
    number = gr.Number(label=label, minimum=minimum, maximum=maximum, value=value, step=step,
                       precision=precision, min_width=min_width)
    return number


def gr_val(val):
    return gr.Number(value=val, visible=False)


def apply_listener(apply_button, apply_func, plot_func, reseted_state, apply_inputs, apply_outputs,
                   plot_inputs, plot_outputs):
    def enable_button(value):
        button = gr.Button(value=value, interactive=True)
        return button

    def disable_button(value):
        button = gr.Button(value=value, interactive=False)
        return button, None

    listener = apply_button.click(disable_button, [apply_button], [apply_button, reseted_state])
    listener = listener.then(apply_func, apply_inputs, apply_outputs, show_progress="minimal")
    listener = listener.then(plot_func, plot_inputs + [gr_val(0)], plot_outputs, show_progress="minimal")
    listener = listener.then(plot_func, plot_inputs + [gr_val(1)], plot_outputs, show_progress="minimal")
    listener = listener.then(plot_func, plot_inputs + [gr_val(2)], plot_outputs, show_progress="minimal")
    listener = listener.then(enable_button, [apply_button], apply_button)

    return


def transform_block():
    x_state = gr.State(value=None)
    x_pdf_state = gr.State(value=None)
   
    title = "Demo 1 - Random Variable Transform In DPM"
    with gr.Accordion(label=title, elem_classes="first_demo", elem_id="demo_1"):
        with gr.Group(elem_classes="normal"):
            with gr.Row():
                init_seed = gr.Number(label="random seed", value=100, minimum=0, step=1)
                shrink_alpha = gr.Slider(label="alpha of linear transform", value=0.7, minimum=0.3, maximum=0.999, step=0.001)
                conv_alpha = gr.Slider(label="alpha of add noises", value=0.995, minimum=0.3, maximum=0.999, step=0.001)
                gr_empty_space(10)
                gr_empty_space(5)

            with gr.Row():
                inp_plot = gr.Plot(label="input random variable's pdf", show_label=False)
                shrink_plot = gr.Plot(label="pdf after linear transform", show_label=False)
                conv_plot = gr.Plot(label="pdf after add noises", show_label=False)
                shrink_conv_plot = gr.Plot(label="pdf after linear transform and add noises", show_label=False)
                gr_empty_space(5)
    
    init_inputs = [init_seed, shrink_alpha, conv_alpha]
    init_outputs = [inp_plot, x_state, x_pdf_state, shrink_plot, conv_plot, shrink_conv_plot]
    init_seed.change(init_change, init_inputs, init_outputs, show_progress="minimal")

    shrink_inputs = [x_state, x_pdf_state, shrink_alpha, conv_alpha]
    shrink_outputs = [shrink_plot, shrink_conv_plot]
    shrink_alpha.change(shrink_change, shrink_inputs, shrink_outputs, show_progress="minimal")

    conv_inputs = [x_state, x_pdf_state, shrink_alpha, conv_alpha]
    conv_outputs = [conv_plot, shrink_conv_plot]
    conv_alpha.change(conv_change, conv_inputs, conv_outputs, show_progress="minimal")

    init_param = dict(method=init_change, inputs=init_inputs, outputs=init_outputs)
    return init_param


def cond_prob_block():
    x_state = gr.State(value=None)
    x_pdf_state = gr.State(value=None)

    z_state = gr.State(value=None)
    xcz_pdf_state = gr.State(value=None)

    title = "Demo 2 - Likelihood and Posterior of Transform"
    with gr.Accordion(label=title, elem_classes="first_demo", elem_id="demo_2"):
        with gr.Group(elem_classes="normal"):
            with gr.Row():
                seed = gr_number("random seed", 0, 1E6, 100, 1, 0, min_width=80)
                alpha = gr_number("alpha", 0.001, 0.999, 0.98, 0.001, 3, min_width=80)
                cond_val = gr.Slider(label="fixed condition value", value=0.2, minimum=g_st, maximum=g_et, step=0.1)
                gr_empty_space(5)
                gr_empty_space(5)

            with gr.Row():
                inp_plot = gr.Plot(label="input variable's pdf", min_width=80, show_label=False)
                out_plot = gr.Plot(label="output variable's pdf", min_width=80, show_label=False)
                forward_cond_plot = gr.Plot(label="forward conditional pdf", min_width=80, show_label=False)
                backward_cond_plot = gr.Plot(label="backward conditional pdf", min_width=80, show_label=False)
                fixed_cond_plot = gr.Plot(label="backward fixed conditional pdf", min_width=80, show_label=False)
            
    init_inputs = [seed, alpha, cond_val]
    init_outputs = [x_state, x_pdf_state, z_state, xcz_pdf_state, inp_plot, out_plot,
                    forward_cond_plot, backward_cond_plot, fixed_cond_plot]
    seed.change(cond_prob_init_change, init_inputs, init_outputs, show_progress="minimal")

    alpha_inputs = [x_state, x_pdf_state, alpha, cond_val]
    alpha_outputs = [z_state, xcz_pdf_state, out_plot, forward_cond_plot, backward_cond_plot, fixed_cond_plot]
    alpha.change(cond_prob_alpha_change, alpha_inputs, alpha_outputs, show_progress="minimal")
    
    cond_inputs = [x_state, x_pdf_state, z_state, xcz_pdf_state, alpha, cond_val]
    cond_outputs = [backward_cond_plot, fixed_cond_plot]
    cond_val.change(cond_prob_cond_change, cond_inputs, cond_outputs, show_progress="minimal")

    init_param = dict(method=cond_prob_init_change, inputs=init_inputs, outputs=init_outputs)

    return init_param


def forward_block(seq_info_state):
    x_state = gr.State(value=None)
    x_pdf_state = gr.State(value=None)
    
    plot_state = gr.State(value=None)
    
    title = "Demo 3.1 - Transform To Normal Distribution Iteratively"
    with gr.Accordion(label=title, elem_classes="first_demo", elem_id="demo_3_1"):
        with gr.Group(elem_classes="normal"):
            with gr.Row():
                seed = gr_number("random seed", 0, 1E6, 100, 1, 0, min_width=80)
                st_alpha = gr_number("start alpha", 0.001, 0.999, 0.98, 0.001, 3, min_width=80)
                et_alpha = gr_number("end alpha", 0.001, 0.999, 0.98, 0.001, 3, min_width=80)
                step = gr.Slider(label="step", value=7, minimum=1, maximum=15, step=1, min_width=80)
                apply_button = gr.Button(value="apply", min_width=80)
        
            node_plot = gr.Plot(label="latent variable's pdf", show_label=False)
            with gr.Accordion("posterior pdf", elem_classes="second"):
                cond_plot = gr.Plot(show_label=False)
 
    apply_inputs = [x_state, x_pdf_state, st_alpha, et_alpha, step]
    apply_outputs = [seq_info_state, plot_state]
    plot_outputs = [node_plot, cond_plot]
    apply_listener(apply_button, forward_seq_apply, forward_plot_part,
                   seq_info_state, apply_inputs, apply_outputs, [plot_state], plot_outputs)
    
    init_outputs = [x_state, x_pdf_state, node_plot, cond_plot]
    seed.change(forward_init_change, inputs=[seed], outputs=init_outputs, show_progress="minimal")
    
    init_param = dict(method=forward_init_change, inputs=[seed], outputs=init_outputs)
    
    return init_param


def backward_block(seq_info_state):
    plot_state = gr.State(value=None)
    placeholder = gr.State(value=None)

    title = "Demo 3.2 - Recover From Normal Distribution Iteratively"
    with gr.Accordion(label=title, elem_classes="first_demo", elem_id="demo_3_2"):
        with gr.Group(elem_classes="normal"):
            with gr.Row():
                is_forward_pdf = gr.Checkbox(label="forward pdf", value=True)
                is_backward_pdf = gr.Checkbox(label="backward pdf", value=True)
                noise_seed = gr_number("nose random seed", 0, 1E6, 200, 1, 0, min_width=80)
                noise_ratio = gr_number("noise ratio", 0, 1, 0.0, 0.1, 1, min_width=80)
                apply_button = gr.Button(value="apply")

            node_plot = gr.Plot(label="each variable's pdf", show_label=False)

    inputs = [seq_info_state, is_forward_pdf, is_backward_pdf, noise_seed, noise_ratio]
    outputs = [node_plot, plot_state]
    apply_listener(apply_button, backward_seq_apply, backward_plot_part,
                   placeholder, inputs, outputs, [plot_state], [node_plot])
    return


def fit_posterior_block(seq_info_state):
    plot_state = gr.State(value=None)
    placeholder = gr.State(value=None)

    title = "Demo 3.3 - Fitting Posterior with Conditional Gaussian Model"
    with gr.Accordion(label=title, elem_classes="first_demo", elem_id="demo_3_3"):
        with gr.Group(elem_classes="normal"):
            with gr.Row():
                info = "show forward pdf"
                is_forward_pdf = gr.Checkbox(label="forward pdf", info=info, value=True)
                info = "show origin backward pdf"
                is_backward_pdf = gr.Checkbox(label="backward pdf", info=info, value=False)
                info = "show backward pdf after fitting posterior with conditonal Gaussian"
                is_show_pos = gr.Checkbox(label="fitted posterior", info=info, value=True)
                apply_button = gr.Button(value="apply")

            node_plot = gr.Plot(label="each variable's pdf", show_label=False)
            with gr.Accordion("fitted posterior's pdf", elem_classes="second"):
                cond_plot = gr.Plot(show_label=False)

    inputs = [seq_info_state, is_forward_pdf, is_backward_pdf]
    outputs = [node_plot, cond_plot, plot_state]
    apply_listener(apply_button, fit_and_backward_apply, fit_plot_part,
                   placeholder, inputs, outputs, [plot_state, is_show_pos], [node_plot, cond_plot])
      
    return


def contraction_block():
    x_state = gr.State(value=None)
    x_pdf_state = gr.State(value=None)
    z_state = gr.State(value=None)
    xcz_pdf_state = gr.State(value=None)
    zt_state = gr.State(value=None)
    zt_pdf_state = gr.State(value=None)
    
    plot_state = gr.State(value=None)
    placeholder = gr.State(value=None)

    ctr_title = "Demo 4.1 - Posterior Transform is a Contraction Mapping"
    with gr.Accordion(label=ctr_title, elem_classes="first_demo", elem_id="demo_4_1"):
        with gr.Row(elem_classes="normal"):
            with gr.Column(scale=3):
                with gr.Group():
                    with gr.Row():
                        ctr_init_seed = gr_number("random seed", 0, 1E6, 100, 1, 0, min_width=80)
                        ctr_alpha = gr_number("alpha", 0.001, 0.999, 0.95, 0.001, 3, min_width=80)
                        lambda_2 = gr_number("second largest eigenvalue", 0, 0, 1.0, 0.0001, 4, min_width=80)

                    with gr.Row():
                        inp_plot = gr.Plot(label="input variable pdf", min_width=80, show_label=False)
                        pos_plot = gr.Plot(label="posterior pdf", min_width=80, show_label=False)
                        out_plot = gr.Plot(label="output variable pdf", min_width=80, show_label=False)

            with gr.Column(scale=2):
                with gr.Group():
                    with gr.Row():
                        change_inputs_seed = gr.Button(value="change inputs seed")
                        two_inputs_seed = gr_number("two inputs random seed", 0, 1E9, 100, 1, 0)
                    inp_out_plot = gr.Plot(label="input and output pdf of inverse transform", show_label=False)
    
    fixed_title = "Demo 4.2 - Posterior Transform Have a Converging Point"
    with gr.Accordion(label=fixed_title, elem_classes="first_demo", elem_id="demo_4_2"):
        with gr.Group(elem_classes="normal"):
            with gr.Row():
                fixed_point_seed = gr_number("input seed", 0, 1E6, 200, 1, 0, min_width=80)
                iterate_number = gr_number("iterate number", 0, 1E6, 500, 1, 0, min_width=80)
                is_show_pow = gr.Checkbox(label="show power matrix", value=True)
                fixed_iterate_btn = gr.Button(value="apply iteration transform")
                gr_empty_space(5)
                gr_empty_space(5)
            fixed_point_plot = gr.Plot(label="result of iteration of inverse transform", show_label=False)
            with gr.Accordion("power matrix of posterior", elem_classes="second"):
                power_mat_plot = gr.Plot(show_label=False)
    
    ctr_init_inputs = [ctr_init_seed, ctr_alpha, two_inputs_seed]
    ctr_init_outputs = [inp_plot, x_state, x_pdf_state, pos_plot, out_plot, z_state, xcz_pdf_state, inp_out_plot, lambda_2]
    ctr_init_seed.change(contraction_init_change, ctr_init_inputs, ctr_init_outputs, show_progress="minimal")
    
    ctr_alpha_inputs = [x_state, x_pdf_state, ctr_alpha, two_inputs_seed]
    ctr_alpha_outputs = [pos_plot, out_plot, z_state, xcz_pdf_state, inp_out_plot, lambda_2]
    ctr_alpha.change(contraction_alpha_change, ctr_alpha_inputs, ctr_alpha_outputs, show_progress="minimal")
    
    ctr_apply_inputs, ctr_apply_outputs = [x_state, x_pdf_state, xcz_pdf_state, two_inputs_seed], [inp_out_plot]
    two_inputs_seed.change(contraction_apply, ctr_apply_inputs, ctr_apply_outputs, show_progress="minimal")
    change_inputs_seed.click(change_two_inputs_seed, None, two_inputs_seed, show_progress="minimal")

    fixed_init_inputs = [fixed_point_seed, x_state, x_pdf_state]
    fixed_init_outputs = [fixed_point_plot, zt_state, zt_pdf_state, power_mat_plot]
    fixed_point_seed.change(fixed_point_init_change, fixed_init_inputs, fixed_init_outputs, show_progress="minimal")
    
    iterate_inputs = [x_state, x_pdf_state, zt_state, zt_pdf_state, xcz_pdf_state, iterate_number, is_show_pow]
    iterate_outputs = [fixed_point_plot, power_mat_plot, plot_state]
    plot_outputs = [fixed_point_plot, power_mat_plot]
    apply_listener(fixed_iterate_btn, fixed_point_apply_iterate, fixed_plot_part, placeholder,
                   iterate_inputs, iterate_outputs, [plot_state], plot_outputs)
    
    ctr_init_param = dict(method=contraction_init_change, inputs=ctr_init_inputs, outputs=ctr_init_outputs)
    fixed_init_param = dict(method=fixed_point_init_change, inputs=fixed_init_inputs, outputs=fixed_init_outputs)
    return ctr_init_param, fixed_init_param


def md_header_block():
    gr.Markdown(""" </br> </br>
        <center>
            <h1 style="display:block">
                Understanding Diffusion Probability Model<span style='color: orange'>&nbsp;Interactively </span>
            </h1>
        </center>
        </br> </br> </br>""")
    return


def run_app():
     
    with gr.Blocks(css=g_css, head=js_head, js=js_load) as demo:
        seq_info_state = gr.State(value=None)
        
        # this is needed for offline render markdown in order to import katex.css
        gr.Markdown("$$ $$", visible=False)
        
        md_header_block()
        
        md_introduction_block()

        md_transform_block()
 
        rets = transform_block()
        trans_param = rets

        md_likelihood_block()

        md_posterior_block()

        rets = cond_prob_block()
        cond_param = rets

        md_forward_process_block()

        rets = forward_block(seq_info_state)
        fore_param = rets

        md_backward_process_block()

        backward_block(seq_info_state)

        md_fit_posterior_block()

        fit_posterior_block(seq_info_state)

        md_posterior_transform_block()

        rets = contraction_block()
        ctr_param, fixed_param = rets
        
        md_deconvolution_block()

        md_cond_kl_block()
        
        md_approx_gauss_block()
        
        md_non_expanding_block()
        
        md_stationary_block()
        
        md_reference_block()
        
        md_about_block()

        gr.Markdown("<div><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br><br></div>", visible=True)

        # running initiation consecutively because of the bug of multithreading rendering mathtext in matplotlib
        demo.load(trans_param["method"], trans_param["inputs"], trans_param["outputs"], show_progress="minimal").\
            then(cond_param["method"], cond_param["inputs"], cond_param["outputs"], show_progress="minimal"). \
            then(fore_param["method"], fore_param["inputs"], fore_param["outputs"], show_progress="minimal"). \
            then(ctr_param["method"], ctr_param["inputs"], ctr_param["outputs"], show_progress="minimal"). \
            then(fixed_param["method"], fixed_param["inputs"], fixed_param["outputs"], show_progress="minimal")
     
    demo.launch(allowed_paths=["/"])
        
    return


def gtx():
     
    with gr.Blocks(css=g_css, head=js_head, js=js_load) as demo:
        gr.Markdown("$$ $$", visible=False)
        
        md_introduction_block()

        md_transform_block()

        md_likelihood_block()

        md_posterior_block()
        
        md_forward_process_block()
        
        md_backward_process_block()
        
        md_fit_posterior_block()
        
        md_posterior_transform_block()
        
        md_deconvolution_block()
        
        md_reference_block()
        
        md_about_block()
        
    demo.queue()
    demo.launch(allowed_paths=["/"])
    return
    

if __name__ == "__main__":
    run_app()
    # gtx()