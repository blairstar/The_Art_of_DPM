
import gradio as gr

from RenderMarkdownZh import md_introduction_zh, md_transform_zh, md_likelihood_zh, md_posterior_zh
from RenderMarkdownZh import md_forward_process_zh, md_backward_process_zh, md_fit_posterior_zh
from RenderMarkdownZh import md_posterior_transform_zh, md_deconvolution_zh, md_reference_zh, md_about_zh

from RenderMarkdownEn import md_introduction_en, md_transform_en, md_likelihood_en, md_posterior_en
from RenderMarkdownEn import md_forward_process_en, md_backward_process_en, md_fit_posterior_en
from RenderMarkdownEn import md_posterior_transform_en, md_deconvolution_en, md_reference_en, md_about_en


def md_introduction_block(md_type="offline"):
    if md_type == "offline":
        gr.Accordion(label="0. Introduction", elem_classes="first_md", elem_id="introduction")
    elif md_type == "zh":
        md_introduction_zh()
    elif md_type == "en":
        md_introduction_en()
    else:
        raise NotImplementedError
    return


def md_transform_block(md_type="offline"):
    if md_type == "offline":
        gr.Accordion(label="1. How To Transform", elem_classes="first_md", elem_id="transform")
    elif md_type == "zh":
        md_transform_zh()
    elif md_type == "en":
        md_transform_en()
    else:
        raise NotImplementedError
    return


def md_likelihood_block(md_type="offline"):
    if md_type == "offline":
        gr.Accordion(label="2. Likelihood of The Transform", elem_classes="first_md", elem_id="likelihood")
    elif md_type == "zh":
        md_likelihood_zh()
    elif md_type == "en":
        md_likelihood_en()
    else:
        raise NotImplementedError
    return


def md_posterior_block(md_type="offline"):
    if md_type == "offline":
        gr.Accordion(label="3. Posterior of The Transform", elem_classes="first_md", elem_id="posterior")
    elif md_type == "zh":
        md_posterior_zh()
    elif md_type == "en":
        md_posterior_en()
    else:
        raise NotImplementedError
    return


def md_forward_process_block(md_type="offline"):
    if md_type == "offline":
        title = "4. Transform Data Distribution To Normal Distribution"
        gr.Accordion(label=title, elem_classes="first_md", elem_id="forward_process")
    elif md_type == "zh":
        md_forward_process_zh()
    elif md_type == "en":
        md_forward_process_en()
    else:
        raise NotImplementedError
    return


def md_backward_process_block(md_type="offline"):
    if md_type == "offline":
        title = "5. Restore Data Distribution From Normal Distribution"
        gr.Accordion(label=title, elem_classes="first_md", elem_id="backward_process")
    elif md_type == "zh":
        md_backward_process_zh()
    elif md_type == "en":
        md_backward_process_en()
    else:
        raise NotImplementedError
    return


def md_fit_posterior_block(md_type="offline"):
    if md_type == "offline":
        title = "6. Fitting Posterior With Conditional Gaussian Model"
        gr.Accordion(label=title, elem_classes="first_md", elem_id="fit_posterior")
    elif md_type == "zh":
        md_fit_posterior_zh()
    elif md_type == "en":
        md_fit_posterior_en()
    else:
        raise NotImplementedError
    return


def md_posterior_transform_block(md_type="offline"):
    if md_type == "offline":
        gr.Accordion(label="7. Posterior Transform", elem_classes="first_md", elem_id="posterior_transform")
    elif md_type == "zh":
        md_posterior_transform_zh()
    elif md_type == "en":
        md_posterior_transform_en()
    else:
        raise NotImplementedError
    return


def md_deconvolution_block(md_type="offline"):
    if md_type == "offline":
        title = "8. Can the data distribution be restored by deconvolution?"
        gr.Accordion(label=title, elem_classes="first_md", elem_id="deconvolution")
    elif md_type == "zh":
        md_deconvolution_zh()
    elif md_type == "en":
        md_deconvolution_en()
    else:
        raise NotImplementedError
    return


def md_reference_block(md_type="offline"):
    if md_type == "offline":
        gr.Accordion(label="Reference", elem_classes="first_md", elem_id="reference")
    elif md_type == "zh":
        md_reference_zh()
    elif md_type == "en":
        md_reference_en()
    else:
        raise NotImplementedError
    return


def md_about_block(md_type="offline"):
    if md_type == "offline":
        gr.Accordion(label="About", elem_classes="first_md", elem_id="about")
    elif md_type == "zh":
        md_about_zh()
    elif md_type == "en":
        md_about_en()
    else:
        raise NotImplementedError
    return
