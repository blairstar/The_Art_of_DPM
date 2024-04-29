
import os
import random
import time

import numpy as np
import pandas as pd
import gradio as gr
import copy
import scipy
import scipy.signal
from scipy.stats import norm
import matplotlib
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from scipy.optimize import curve_fit

import multiprocessing
from multiprocessing import Pool, Queue, Manager

plt.rcParams['figure.constrained_layout.use'] = True
plt.rcParams['figure.max_open_warning'] = 10
matplotlib.rcParams['interactive'] = False


g_st, g_et, g_num = -2.3, 2.3, 460
g_res = (g_et-g_st)/g_num
g_fw, g_fh = 3, 3.2


###################################################################################
# common function
###################################################################################

def rs(str_label):
    return str_label.replace("z_{0}", "x").replace("z_0", "x")


def set_axis(axis, x_range, y_range, x_label, y_label):
    matplotlib.rcParams.update({'font.size': 10, "axes.linewidth": 0.5, "lines.linewidth": 0.7, "figure.dpi": 100})
    
    if x_range is not None:
        axis.set_xlim(*x_range)
    if y_range is not None:
        axis.set_ylim(*y_range)
    if x_label is not None:
        axis.set_xlabel(x_label)
    if y_label is not None:
        axis.set_ylabel(y_label)
        
    axis.xaxis.set_major_locator(plt.MultipleLocator(1))
    st, et = x_range[0]//0.2*0.2, x_range[1]//0.2*0.2
    count = int((et - st)/0.2)
    axis.set_xticks(np.linspace(st, et, count+1), minor=True)

    return


def plot_pdf(x, x_pdf, max_y=3.2, title=None, titlesize=10,
             label=None, xlabel="domain", ylabel="pdf", style="solid", color="blue"):
    fig = plt.figure(figsize=(g_fw, g_fh))
    ax = fig.add_subplot(111)
    axis_pdf(ax, x, x_pdf, max_y, title, titlesize, label, xlabel, ylabel, style, color)
    return fig


def plot_2d_pdf(x, y, pdf, cond_val=None, label=None, title=None, titlesize=10, xlabel="x", ylabel="y"):
    fig = plt.figure(figsize=(g_fw, g_fh))
    ax = fig.add_subplot(111)
    axis_2d_pdf(ax, x, y, pdf, cond_val, title, titlesize, label, xlabel, ylabel)
    return fig


def axis_pdf(ax, x, x_pdf, max_y=3.2, title=None, titlesize=10,
             label=None, xlabel="domain", ylabel="pdf", style="solid", color="blue"):
    set_axis(ax, (x[0], x[-1]), (0, max_y), xlabel, ylabel)
    ax.plot(x, x_pdf, label=label, color=color, linestyle=style)
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    ax.legend()
    return


def axis_2d_pdf(ax, x, y, pdf, cond_val=None, title=None, titlesize=10, label=None, xlabel="x", ylabel="y"):
    set_axis(ax, (x[0], x[-1]), (y[0], y[-1]), xlabel, ylabel)
    ax.contourf(x, y, pdf, label=label)
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if cond_val is not None:
        ax.plot([cond_val, cond_val], [y[-1], y[0]], color="orange")
    ax.legend()
    return


def add_random_noise(x_pdf, noise_ratio, seed, st, et, num, res):
    _, noise_pdf = init_x_pdf(st, et, num, seed=seed)
    z_pdf = (1-noise_ratio)*x_pdf + noise_ratio*noise_pdf
    z_pdf = z_pdf/(res*z_pdf.sum())
    return z_pdf


def power_range(st, et, num, coeff=2):
    roi_nodes = st + np.ceil((np.linspace(0, 1, num=num) ** coeff) * (et - st))
    roi_nodes = roi_nodes.astype(int)

    for ii in range(1, len(roi_nodes)):
        if roi_nodes[ii] <= roi_nodes[ii - 1]:
            roi_nodes[ii] = roi_nodes[ii - 1] + 1
        roi_nodes[ii] = int(roi_nodes[ii])

    return list(roi_nodes)


def init_x_pdf(st, et, num, modal_count=16, shape_type=0, seed=200):
    rg = np.random.RandomState(int(seed))
    
    C = modal_count
    res = (et - st) / num

    if shape_type == 0:
        mean_st, mean_et = -1.1, 1.1
        std_st, std_et = 0.03, 0.20
    elif shape_type == 1:
        mean_st, mean_et = -1.5, 1.5
        std_st, std_et = 0.01, 0.09
    elif shape_type == 2:
        mean_st, mean_et = -1.5, 1.5
        std_st, std_et = 0.05, 0.35
    else:
        mean_st, mean_et = -0.8, 0.8
        std_st, std_et = 0.05, 0.35

    mean = mean_st + rg.random(C) * (mean_et - mean_st)
    std = std_st + rg.random(C) * (std_et - std_st)
    weight = 1 + rg.random(C) * 10
    weight = weight / weight.sum()

    x = np.linspace(st, et, num + 1, dtype=np.float64)
    x_pdf = np.zeros_like(x, dtype=np.float64)

    for i in range(C):
        # print("%+0.5f___%+0.5f___%+0.5f" % (mean[i], std[i], weight[i]))
        x_pdf += weight[i] * norm.pdf(x, mean[i], std[i])
    x_pdf += 1E-8
    x_pdf = x_pdf / (x_pdf * res).sum()  # normalized to 1

    return x, x_pdf


def forward_next_pdf(x, x_pdf, alpha, res):
    '''
        x       :   input domain
        x_pdf   :   input pdf of continual variable
        res     :   resolution of x's domain

        Two ways to understand normalizing to 1:
            convert to discrete variable and summarize
            Approximate integral for continual variable
    '''

    if np.isclose(alpha, 1.0):
        return x, x_pdf, None, None, None

    y = copy.deepcopy(x)
    xy_pdf = np.zeros([*x.shape, *y.shape], dtype=np.float64)
    for i in range(len(x)):
        p_x = x_pdf[i]
        mu = x[i] * np.sqrt(alpha)
        std = np.sqrt(1 - alpha)
        p_y__x = norm.pdf(y, mu, std)
        
        p_y__x = p_y__x/(p_y__x*res).sum()
        
        # this will cause posterior distortion in the near zero area
        # p_y__x += 1E-8
        # p_y__x = p_y__x / (p_y__x * res).sum()  # normalize to 1

        xy_pdf[i] = p_x * p_y__x

    xy_pdf = xy_pdf / (xy_pdf * res * res).sum()  # normalize to 1
    y_pdf = (xy_pdf * res).sum(axis=0)

    xcy_pdf = xy_pdf / (y_pdf[None, :] + 1E-10)
    ycx_pdf = xy_pdf / (x_pdf[:, None] + 1E-10)

    return y, y_pdf, xy_pdf, xcy_pdf, ycx_pdf


###################################################################################
# transform block function
###################################################################################

def shrink(x, x_pdf, alpha, st, res):
    '''
    x                   :   input domain
    x_pdf               :   input pdf of continual variable
    function            :   y = sqrt(\alpha) * x
    inverse function    :   x = y / sqrt(\alpha)
    derivative          :   y'= sqrt(\alpha)
    '''

    # y's domain is the sample as x
    y = copy.deepcopy(x)
    shrink_pdf = np.zeros_like(x_pdf, dtype=np.float64)
    sqrt_alpha = np.sqrt(alpha)

    for i in range(len(y)):
        # get corresponding x by inverse function
        idx = int((y[i] / sqrt_alpha - st) / res)
        if idx < 0 or idx >= len(x_pdf):
            continue

        # scale with the reciprocal of derivative of y
        shrink_pdf[i] = (1 / sqrt_alpha) * x_pdf[idx]

    return shrink_pdf


def conv(x, x_pdf, alpha, res):
    # gauss_pdf is continual random variable pdf
    gauss_pdf = norm.pdf(x, 0, np.sqrt(1 - alpha))

    # convert to discrete probability by multiplying with res, and convert back to continual by dividing res
    out_pdf = scipy.signal.convolve(x_pdf * res, gauss_pdf * res, "same") / res

    return out_pdf


def shrink_conv(x, x_pdf, shrink_alpha, conv_alpha, st, res):
    # linear transform
    shrink_pdf = shrink(x, x_pdf, shrink_alpha, st, res)

    # add independent noises, that is equivalent to convolution
    conv_pdf = conv(x, shrink_pdf, conv_alpha, res)

    return conv_pdf


def plot_init_pdf(seed, st, et, num):
    x, x_pdf = init_x_pdf(st, et, num, shape_type=0, seed=seed)
    fig = plot_pdf(x, x_pdf, label="x", title="input variable's pdf")
    fig.axes[0].title.set_size(9)
   
    return fig, x, x_pdf


def plot_shrink_pdf(x, x_pdf, alpha, st, res):
    if x is None or x_pdf is None:
        return None

    shrink_pdf = shrink(x, x_pdf, alpha, st, res)
    fig = plot_pdf(x, shrink_pdf, label=r"$y=\sqrt{\alpha}x$", title="pdf after linear transform", titlesize=9)
    
    return fig


def plot_conv_pdf(x, x_pdf, alpha, res):
    if x is None or x_pdf is None:
        return None

    conv_pdf = conv(x, x_pdf, alpha, res)
    fig = plot_pdf(x, conv_pdf, label=r"$y=x+\sqrt{1-\alpha}\epsilon$", title="pdf after add noises", titlesize=9)
    
    return fig


def plot_shrink_conv_pdf(x, x_pdf, shrink_alpha, conv_alpha, st, res):
    if x is None or x_pdf is None:
        return None

    shrink_conv_pdf = shrink_conv(x, x_pdf, shrink_alpha, conv_alpha, st, res)
    title = r"pdf after two sub transforms"
    label = r"$y=\sqrt{\alpha_s}x + \sqrt{1-\alpha_e}\epsilon$"
    fig = plot_pdf(x, shrink_conv_pdf, label=label, title=title, titlesize=9)
    
    return fig


def init_change(seed, shrink_alpha, conv_alpha):
    global g_st, g_et, g_num, g_res

    init_fig, x, x_pdf = plot_init_pdf(seed, g_st, g_et, g_num)
    shrink_fig = plot_shrink_pdf(x, x_pdf, shrink_alpha, g_st, g_res)
    conv_fig = plot_conv_pdf(x, x_pdf, conv_alpha, g_res)
    shrink_conv_fig = plot_shrink_conv_pdf(x, x_pdf, shrink_alpha, conv_alpha, g_st, g_res) 

    return init_fig, x, x_pdf, shrink_fig, conv_fig, shrink_conv_fig


def shrink_change(x, x_pdf, shrink_alpha, conv_alpha):
    global g_st, g_et, g_num, g_res

    shrink_fig = plot_shrink_pdf(x, x_pdf, shrink_alpha, g_st, g_res)
    shrink_conv_fig = plot_shrink_conv_pdf(x, x_pdf, shrink_alpha, conv_alpha, g_st, g_res)
    return shrink_fig, shrink_conv_fig


def conv_change(x, x_pdf, shrink_alpha, conv_alpha):
    global g_st, g_et, g_num, g_res

    conv_fig = plot_conv_pdf(x, x_pdf, conv_alpha, g_res)
    shrink_conv_fig = plot_shrink_conv_pdf(x, x_pdf, shrink_alpha, conv_alpha, g_st, g_res)
    return conv_fig, shrink_conv_fig


###################################################################################
# cond prob block function
###################################################################################

def cond_prob_init_change(seed, alpha, cond_val):
    global g_st, g_et, g_num, g_res

    x, x_pdf = init_x_pdf(g_st, g_et, g_num, shape_type=0, seed=seed)
    x_pdf = hijack(seed, x, x_pdf)
    
    fig_x = plot_pdf(x, x_pdf, xlabel="x domain", ylabel="pdf", title="input variable's pdf", titlesize=9)

    outputs = cond_prob_alpha_change(x, x_pdf, alpha, cond_val)
    z, zcx_pdf, fig_z, fig_zcx, fig_xcz, fig_fix_xcz = outputs

    return x, x_pdf, z, zcx_pdf, fig_x, fig_z, fig_zcx, fig_xcz, fig_fix_xcz


def cond_prob_alpha_change(x, x_pdf, alpha, cond_val):
    forward_info = forward_next_pdf(x, x_pdf, alpha, g_res)
    z, z_pdf, xz_pdf, xcz_pdf, zcx_pdf = forward_info
    
    label = r"$z=\sqrt{\alpha}x + \sqrt{1-\alpha}\epsilon$"
    input_title = r"output variable's pdf"
    fore_cond_title = r"forward conditional pdf"
    fig_z = plot_pdf(z, z_pdf, label=label, title=input_title, titlesize=9, xlabel="z domain", ylabel="pdf")
    fig_zcx = plot_2d_pdf(x, z, zcx_pdf.transpose(), label="$q(z|x)$",
                          title=fore_cond_title, titlesize=9, xlabel="x domain(cond)", ylabel="z domain")

    ret_fig = cond_prob_cond_change(x, x_pdf, z, xcz_pdf, alpha, cond_val)
    fig_xcz, fig_fix_xcz = ret_fig

    return z, xcz_pdf, fig_z, fig_zcx, fig_xcz, fig_fix_xcz


def cond_prob_cond_change(x, x_pdf, z, xcz_pdf, alpha, cond_val):
    global g_st, g_et, g_num, g_res

    cond_idx = int((cond_val - g_st) / g_res)
    cond_pdf = xcz_pdf[:, cond_idx]
    
    back_cond_title = "backward conditional pdf"
    fig_xcz = plot_2d_pdf(x, z, xcz_pdf, cond_val, label="$q(x|z)$",
                          title=back_cond_title, xlabel="z domain(cond)", ylabel="x domain")
    fig_xcz.axes[0].title.set_size(9)

    gauss = norm.pdf(x, cond_val / np.sqrt(alpha), np.sqrt((1 - alpha) / alpha))
    
    fixed_back_cond_title = "posterior with fixed condition"
    fig_fix_xcz = plt.figure(figsize=(g_fw, g_fh))
    ax = fig_fix_xcz.add_subplot(111)
    axis_pdf(ax, x, gauss, max_y=5, label="$gauss$", style="dashed", color="green")
    axis_pdf(ax, x, x_pdf, max_y=5, label="$q(x)$", style="dashed", color="blue")
    axis_pdf(ax, x, cond_pdf, max_y=5, label="$q(x|z=%s)$" % cond_val,
             title=fixed_back_cond_title, titlesize=9, xlabel="x domain", color="orange")
    handles, labels = ax.get_legend_handles_labels()
    ax.add_artist(ax.legend(handles[:2], labels[:2], handlelength=0.8, loc="upper left"))
    ax.add_artist(ax.legend(handles[2:], labels[2:], handlelength=0.8, loc="upper right"))

    return fig_xcz, fig_fix_xcz


###################################################################################
# forward block function
###################################################################################


def plot_first_pdf(x, x_pdf, ax):
    title, label = r"origin var pdf", r"forward q(x)",
    xlabel = rs(r"x domain")
    axis_pdf(ax, x, x_pdf, title=title, label=label, xlabel=xlabel, ylabel="pdf", color="blue")
    ax.legend(handlelength=1.2, labels=[label])
    return


def forward_init_change(seed):
    global g_st, g_et, g_num, g_res
    
    x, x_pdf = init_x_pdf(g_st, g_et, g_num, seed=seed)
    x_pdf = hijack(seed, x, x_pdf)
    
    fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(8 * g_fw, 1 * g_fh))
    axes = axes.flatten()
    
    plot_first_pdf(x, x_pdf, axes[0])
    
    return x, x_pdf, fig, None


def plot_forward_pdf(axes, seq_info, color, pidx=-1):
    count = len(seq_info)
    step = int(count/3+1)

    if pidx >= 0:
        st, et = pidx*step, (pidx+1)*step
        seq_info = seq_info[st:et]
        
    for info in seq_info:
        _, _, nz, nz_pdf, _, cidx, nidx, alpha = info
        if nidx == 0:
            title, label = "origin var pdf", r"forward $q(x)$",
        else:
            title, label = rs(r"$q(z_{%d})\ \alpha=%0.3f$"%(nidx, alpha)), r"forward $q(z_{%d})$"%nidx
            
        xlabel = rs(r"$z_{%d}\ domain$"%nidx)
        axis_pdf(axes[nidx], nz, nz_pdf, title=title, label=label, xlabel=xlabel, ylabel="pdf", color=color)
        axes[nidx].legend(handlelength=1.2)
    
        if nidx == (count-1):
            axes[count-1].plot(nz, norm.pdf(nz, 0, 1), label=r"$\mathcal{N}\/(0, 1)$", color="green")
            axes[count-1].legend()
    
    return


def plot_backward_pdf(axes, fore_seq_info, back_seq_info, label_prefix, res, color, pidx=-1):
    count = len(fore_seq_info)
    step = int(count/3 + 1)
    
    if pidx >= 0:
        st, et = (2-pidx)*step, (2-pidx+1)*step        # reverse
        fore_seq_info, back_seq_info = fore_seq_info[st:et], back_seq_info[st:et]
    
    for fore_info, back_info in zip(fore_seq_info, back_seq_info):
        fore_nz_pdf, back_nz_pdf = fore_info[3], back_info[3]
        nz, nidx = fore_info[2], fore_info[6]
         
        div = jensenshannon(back_nz_pdf*res, fore_nz_pdf*res)
        
        name = r"$\mathcal{N}\/(0,1)$" if nidx == count-1 else "revert"   # specific name at end point
        label = rs(label_prefix + name + " div=%0.2f"%div)
        xlabel = rs(r"$z_{%d}\ domain$" % nidx)
        axis_pdf(axes[nidx], nz, back_nz_pdf, label=label, xlabel=xlabel, ylabel="pdf", color=color)
        axes[nidx].legend(handlelength=1.2)
    
    return


def plot_backward_cond_pdf(axes, seq_info, reverse=True, pidx=-1):
    count = len(seq_info)
    step = int(count/3+1)

    if pidx >= 0:
        st, et = ((2-pidx)*step, (2-pidx+1)*step) if reverse else (pidx*step, (pidx+1)*step)
        seq_info = seq_info[st:et]

    for info in seq_info:
        cz, cz_pdf, nz, nz_pdf, bc_pdf, cidx, nidx, alpha = info
        if bc_pdf is None:
            continue
        title = rs(r"$q(z_{%d}|z_{%d})\ \alpha=%0.3f$" % (cidx, nidx, alpha))
        xlabel, ylabel = rs(r"$z_{%d}$" % nidx), rs(r"$z_{%d}$" % cidx)
        axis_2d_pdf(axes[nidx], cz, nz, bc_pdf, title=title, xlabel=xlabel, ylabel=ylabel)
    
    return


def get_back_seq_info(ez, ez_pdf, fore_seq_info, res):
    back_seq_info = copy.deepcopy(fore_seq_info)
    count = len(back_seq_info)
    
    nz, nz_pdf = ez, ez_pdf
    for ii in reversed(range(count)):
        bc_pdf = back_seq_info[ii][4]
        if bc_pdf is None:
            back_seq_info[ii][2:4] = nz, nz_pdf
            continue
        cz_pdf = np.matmul(bc_pdf, nz_pdf[:, None]) * res
        cz, cz_pdf = nz, cz_pdf.flatten()
        back_seq_info[ii][:4] = cz, cz_pdf, nz, nz_pdf
        
        nz, nz_pdf = cz, cz_pdf
        
    return back_seq_info


def forward_seq_apply(x, x_pdf, st_alpha, et_alpha, step):
    global g_st, g_et, g_num, g_res
    
    if x_pdf is None:
        return None, None, None, None
    
    alphas = np.linspace(st_alpha, et_alpha, step)
    col_count = 8
    row_count = int(np.ceil((step+1)/8))
     
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(col_count*g_fw, row_count*g_fh))
    axes = axes.flatten()
    pos_fig, pos_axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(col_count*g_fw, row_count*g_fh))
    pos_axes = pos_axes.flatten()

    # plot_first_pdf(x, x_pdf, fig, axes[0])

    seq_info = [[None, None, x, x_pdf, None, -1, 0, None]]
    cz, cz_pdf = x, x_pdf
    for ii, alpha in enumerate(alphas):
        forward_info = forward_next_pdf(cz, cz_pdf, alpha, g_res)
        nz, nz_pdf, joint_pdf, bc_pdf, fc_pdf = forward_info
        
        cidx, nidx = ii, ii+1
        # title, label = r"$q(z_%d)\ \alpha=%0.3f$"%(nidx, alpha), r"$q(z_%d)$"%nidx
        # axis_pdf(axes[nidx], nz, nz_pdf, title=title, label=label, xlabel=r"$z_{%d}\ domain$"%nidx, ylabel="pdf")
        
        # bc_label = rs(r"$q(z_%d|z_%d)\ \alpha=%0.3f$"%(cidx, nidx, alpha))
        # bc_xlabel, bc_ylabel = rs(r"$z_%d$"%nidx), rs(r"$z_%d$"%cidx)
        # axis_2d_pdf(back_axes[nidx], cz, nz, bc_pdf, label=bc_label, xlabel=bc_xlabel, ylabel=bc_ylabel)
        
        seq_info.append([cz, cz_pdf, nz, nz_pdf, bc_pdf, cidx, nidx, alpha])
        cz, cz_pdf = nz, nz_pdf
    
    # plot_forward_pdf(axes, seq_info, "blue")
    # plot_backward_bc_pdf(back_axes, seq_info)
     
    # fig.tight_layout()
    # back_fig.tight_layout()
    
    forward_plot_state = fig, axes, pos_fig, pos_axes, seq_info, g_res, "blue"
    
    return seq_info, forward_plot_state


def forward_plot_part(plot_state, pidx):
    if plot_state is None:
        return None, None
    
    fig, axes, back_fig, pos_axes, seq_info, res, color = plot_state
    
    plot_forward_pdf(axes, seq_info, color, pidx)
    plot_backward_cond_pdf(pos_axes, seq_info, False, pidx)
      
    # fig.tight_layout()
    # back_fig.tight_layout()
    
    return fig, back_fig


def backward_seq_apply(fore_seq_info, is_forward_pdf, is_backward_pdf, noise_seed, noise_ratio):
    global g_st, g_et, g_num, g_res

    if fore_seq_info is None:
        return None, None

    col_count = 8
    step = len(fore_seq_info)-1
    row_count = int(np.ceil((step+1)/8))
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(col_count*g_fw, row_count*g_fh))
    axes = axes.flatten()
    
    x, x_pdf = fore_seq_info[0][2:4]
    
    if is_forward_pdf:
        plot_forward_pdf(axes, fore_seq_info, "blue")

    ez, ez_pdf = fore_seq_info[-1][2], norm.pdf(x, 0, 1)
    
    std_back_seq_info, noise_back_seq_info = None, None
    if is_backward_pdf:
        # plot_backward_pdf(axes, ez, ez_pdf, fore_seq_info, g_res, "std ", color="green")
        std_back_seq_info = get_back_seq_info(ez, ez_pdf, fore_seq_info, g_res)

    if noise_ratio > 0:
        ez_pdf = add_random_noise(ez_pdf, noise_ratio, noise_seed, g_st, g_et, g_num, g_res)
        # plot_backward_pdf(axes, ez, ez_pdf, fore_seq_info, g_res, "noise ", color="red")
        noise_back_seq_info = get_back_seq_info(ez, ez_pdf, fore_seq_info, g_res)

    # fig.tight_layout()
    plot_state = fig, axes, fore_seq_info, std_back_seq_info, noise_back_seq_info, g_res
    
    return fig, plot_state


def backward_plot_part(plot_state, pidx=-1):
    if plot_state is None:
        return None
    
    fig, axes, fore_seq_info, std_back_seq_info, noise_back_seq_info, res = plot_state
    if std_back_seq_info is not None:
        plot_backward_pdf(axes, fore_seq_info, std_back_seq_info, "std ", res, "green", pidx)
    if noise_back_seq_info is not None:
        plot_backward_pdf(axes, fore_seq_info, noise_back_seq_info, "noise ", res, "red", pidx)

    return fig


def fit_pos_with_gauss(idx, x, bc_pdf, queue):
    # bc_pdf = copy.deepcopy(bc_pdf)
    for ii in range(bc_pdf.shape[1]):
        # guess = bc_pdf[:, ii].mean()
        (mu, std), _ = curve_fit(norm.pdf, x, bc_pdf[:, ii], p0=[0, 1])
        bc_pdf[:, ii] = norm.pdf(x, mu, std)
    # queue.put((idx, bc_pdf))
    return bc_pdf


def seq_fit_pos_with_gauss(fore_seq_info):
    fit_seq_info = copy.deepcopy(fore_seq_info)
    
    # queue = Manager().Queue()
    # ls_param = []
    threads = []
    for ii in range(len(fit_seq_info)):
        x, _, _, _, bc_pdf = fit_seq_info[ii][:5]
        if bc_pdf is None:
            continue
        # os.system("echo hihi")
        # thrd = Thread(target=fit_pos_with_gauss, args=(ii, x, bc_pdf, None))
        # threads.append(thrd)
        fit_seq_info[ii][4] = fit_pos_with_gauss(ii, x, bc_pdf, None)
        # ls_param.append((ii, x, bc_pdf, None))
    
    # for thrd in threads:
    #     thrd.start()
    # for thrd in threads:
    #     thrd.join()
    
    # with Pool(6) as pool:
    #     pool.starmap(fit_pos_with_gauss, ls_param)
    # 
    # for ii in range(queue.qsize()):
    #     idx, bc_pdf = queue.get()
    #     seq_info[idx][4] = bc_pdf
    # with WorkerPool(n_jobs=5) as pool:
    #     results = pool.map(fit_pos_with_gauss, ls_param)
    return fit_seq_info


def fit_and_backward_apply(fore_seq_info, is_forward_pdf, is_backward_pdf):
    global g_st, g_et, g_num, g_res
    
    if fore_seq_info is None:
        return None, None, None
    
    col_count = 8
    step = len(fore_seq_info)-1
    row_count = int(np.ceil((step+1) / 8))
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(col_count*g_fw, row_count*g_fh))
    axes = axes.flatten()
    pos_fig, pos_axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(col_count*g_fw, row_count*g_fh))
    pos_axes = pos_axes.flatten()

    x, x_pdf = fore_seq_info[0][2:4]
    # axis_pdf(axes[0], x, x_pdf, title="origin var pdf $q(x)$", label="forward", xlabel="x domain", ylabel="pdf")

    if is_forward_pdf:
        plot_forward_pdf(axes, fore_seq_info, "blue")

    ez, ez_pdf = fore_seq_info[-1][2], norm.pdf(x, 0, 1)

    # axes[step].plot(ez, ez_pdf, label="$\mathcal{N}\/(0, 1)$", color="green")
    
    if is_backward_pdf:
        std_back_seq_info = get_back_seq_info(ez, ez_pdf, fore_seq_info, g_res)
        plot_backward_pdf(axes, fore_seq_info, std_back_seq_info, "std ", g_res, "green")

    fit_back_seq_info = seq_fit_pos_with_gauss(fore_seq_info)
    fit_back_seq_info = get_back_seq_info(ez, ez_pdf, fit_back_seq_info, g_res)
        # plot_backward_pdf(axes, ez, ez_pdf, fit_seq_info, g_res, "fit ", color="orange")
        # plot_backward_bc_pdf(back_axes, seq_info)
        
    # fig.tight_layout()
    # back_fig.tight_layout()
    
    fit_plot_state = fig, axes, pos_fig, pos_axes, fore_seq_info, fit_back_seq_info, g_res
    
    return fig, pos_fig, fit_plot_state


def fit_plot_part(plot_state, is_show_pos, pidx=-1):
    if plot_state is None:
        return None, None
    
    fig, axes, back_fig, back_axes, fore_seq_info, fit_back_seq_info, res = plot_state
    plot_backward_pdf(axes, fore_seq_info, fit_back_seq_info, "fit ", res, "orange", pidx)
    if is_show_pos:
        plot_backward_cond_pdf(back_axes, fit_back_seq_info, True, pidx)
        # back_fig.tight_layout()
        
    return fig, back_fig


###################################################################################
# contraction block function
###################################################################################

def contraction_init_change(seed, alpha, two_inputs_seed):
    global g_st, g_et, g_num, g_res
    
    rg = np.random.RandomState(int(seed))
    shape_type = rg.randint(0, 4)
    
    x, x_pdf = init_x_pdf(g_st, g_et, g_num, shape_type=shape_type, seed=seed)
    x_pdf = hijack(seed, x, x_pdf)
    
    # test
    x_pdf[x_pdf < 0.01] = 0
    
    x_pdf = x_pdf / (x_pdf * g_res).sum()  # normalized to 1
    fig = plot_pdf(x, x_pdf, title="input variable pdf", titlesize=9)
    
    info = contraction_alpha_change(x, x_pdf, alpha, two_inputs_seed)
    fig_xcz, fig_z, z, xcz_pdf, fig_inp_out = info
    
    return fig, x, x_pdf, fig_xcz, fig_z, z, xcz_pdf, fig_inp_out


def contraction_alpha_change(x, x_pdf, alpha, two_inputs_seed):
    global g_st, g_et, g_num, g_res
    
    forward_info = forward_next_pdf(x, x_pdf, alpha, g_res)
    z, z_pdf, xz_pdf, xcz_pdf, zcx_pdf = forward_info

    label = r"$z=\sqrt{\alpha}x + \sqrt{1-\alpha}\epsilon$"
    z_title = r"output variable pdf"
    xcz_title = r"posterior pdf"
    fig_z = plot_pdf(z, z_pdf, label=label, title=z_title, titlesize=9, xlabel="z domain", ylabel="pdf")
    fig_xcz = plot_2d_pdf(x, z, xcz_pdf, None, label="$q(x|z)$",
                          title=xcz_title, titlesize=9, xlabel="z domain(cond)", ylabel="x domain")

    fig_inp_out = contraction_apply(x, x_pdf, xcz_pdf, two_inputs_seed)

    return fig_xcz, fig_z, z, xcz_pdf, fig_inp_out


def change_two_inputs_seed():
    seed = random.randint(0, 1E6)
    return seed


def contraction_apply(x, x_pdf, bc_pdf, seed):
    global g_st, g_et, g_num, g_res

    rg = np.random.RandomState(int(seed))
    
    modals = [1, 2, 8, 12, 16, 16, 16, 16, 16, 20]
    count1, count2 = rg.choice(modals), rg.choice(modals)
    
    seed1, seed2 = rg.randint(0, 1E6, 2)
    shape1, shape2 = rg.randint(0, 4, 2)

    z1, z1_pdf = init_x_pdf(g_st, g_et, g_num, count1, shape_type=shape1, seed=seed1)
    z2, z2_pdf = init_x_pdf(g_st, g_et, g_num, count2, shape_type=shape2, seed=seed2)

    div_z = jensenshannon(z1_pdf*g_res, z2_pdf*g_res)
    
    x1_pdf = np.matmul(bc_pdf, z1_pdf[:, None]) * g_res
    x2_pdf = np.matmul(bc_pdf, z2_pdf[:, None]) * g_res
    x1_pdf, x2_pdf = x1_pdf.flatten(), x2_pdf.flatten()

    div_x = jensenshannon(x1_pdf*g_res, x2_pdf*g_res)
    
    div_in_label, div_out_label = r"$div_{in}=%0.3f$"%div_z, r"$div_{out}=%0.3f$"%div_x

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(2*g_fw, 1*g_fh))
    
    axis_pdf(axes[0], z1, z1_pdf, max_y=3.8, label="input1",
             title="two random input", titlesize=9, xlabel="z domain", ylabel="pdf", color="orange")
    axis_pdf(axes[0], z2, z2_pdf, max_y=3.8, label="input2", xlabel="z domain", ylabel="pdf", color="green")
    axes[0].plot([], [], label=div_in_label, color="blue")
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].add_artist(axes[0].legend(handles[:2], labels[:2], handlelength=1.0, loc="upper left"))
    axes[0].add_artist(axes[0].legend(handles[2:], labels[2:], handlelength=0, loc="upper right"))

    # axis_pdf(axes[1], x, x_pdf, max_y=3.8, title="two output", titlesize=9, style="dotted", color="blue")
    axis_pdf(axes[1], z1, x1_pdf, max_y=3.8, label="output1",
             title="two output", titlesize=9, xlabel="x domain", ylabel="pdf", color="orange")
    axis_pdf(axes[1], z2, x2_pdf, max_y=3.8, label="output2", xlabel="x domain", ylabel="pdf", color="green")
    axes[1].plot([], [], label=div_out_label, color="blue")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].add_artist(axes[1].legend(handles[:2], labels[:2], handlelength=1.0, loc="upper left"))
    axes[1].add_artist(axes[1].legend(handles[2:], labels[2:], handlelength=0, loc="upper right"))
    
    # fig.tight_layout()
    
    return fig


def fixed_point_init_change(seed, x, x_pdf):
    rg = np.random.RandomState(int(seed))
    
    shape_type = rg.randint(0, 4)
    count = rg.choice([1, 2, 8, 12, 16, 16, 16, 16, 16, 20])

    z, z_pdf = init_x_pdf(g_st, g_et, g_num, modal_count=count, shape_type=shape_type, seed=seed)
    div = jensenshannon(z_pdf*g_res, x_pdf*g_res)
    
    fig, axes = plt.subplots(nrows=1, ncols=8, figsize=(8*g_fw, 1*g_fh))
    axes = axes.flatten()
    axis_pdf(axes[0], x, x_pdf, label="converging pdf", color="blue")
    axis_pdf(axes[0], z, z_pdf, title="random input of inverse transform", label="random input", color="green")
    axes[0].plot([], [], label="div=%0.2f"%div, color="orange")
    axes[0].legend(handlelength=1.2)
    
    # fig.tight_layout()
    return fig, z, z_pdf, None


def matrix_power(in_mat, n):
    if n == 0:
        return np.eye(in_mat.shape[0])

    temp_mat = matrix_power(in_mat, int(n / 2))

    if n % 2 == 0:
        out_mat = np.matmul(temp_mat * 100, temp_mat * 100) / 10000
        out_mat = out_mat / (out_mat.sum(axis=0, keepdims=True) + 1E-9)
        return out_mat
    else:
        out_mat = np.matmul(temp_mat * 100, temp_mat * 100) / 10000
        out_mat = out_mat / (out_mat.sum(axis=0, keepdims=True) + 1E-9)
        out_mat = np.matmul(in_mat * 100, out_mat * 100) / 10000
        out_mat = out_mat / (out_mat.sum(axis=0, keepdims=True) + 1E-9)
        return out_mat


def fixed_point_apply_iterate(x, x_pdf, zt, zt_pdf, xcz_pdf, iterate_num, is_show_pow):
    global g_res
    
    if x_pdf is None or zt_pdf is None or xcz_pdf is None:
        return None, None, None
     
    col_count, max_row_count = 8, 3
    max_ax_count = max_row_count*col_count - 1
    
    ax_count = min(iterate_num, max_ax_count)
    row_count = int(np.ceil((ax_count+1)/col_count))
    fig, axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(col_count*g_fw, row_count*g_fh))
    axes = axes.flatten()
    
    axis_pdf(axes[0], x, x_pdf, label="converging point", color="blue")
    axis_pdf(axes[0], zt, zt_pdf, title="random input", label="random input", color="green")
    
    div = jensenshannon(zt_pdf*g_res, x_pdf*g_res)
    axes[0].plot([], [], label="div=%0.2f"%div, color="green")
    axes[0].legend(handlelength=1.2)
    
    idxs = np.arange(iterate_num).tolist()
    if iterate_num > max_ax_count:
        idxs = np.arange(6).tolist() + power_range(6, iterate_num-1, max_ax_count-6, 2.5)
        
    pow_mats, pdfs = [], []
    for ii, idx in enumerate(idxs):
        pow_idx, ax_idx = idx + 1, ii + 1
        pow_mat = matrix_power(xcz_pdf*g_res, pow_idx)
        pz_pdf = np.matmul(pow_mat, zt_pdf[:, None])
        pz, pz_pdf = zt, pz_pdf.flatten()
        pow_mats.append([pow_mat, pow_idx, ax_idx])
        pdfs.append([x, x_pdf, pz_pdf, pow_idx, ax_idx])
     
    pow_fig, pow_axes = None, None
    if is_show_pow:
        pow_fig, pow_axes = plt.subplots(nrows=row_count, ncols=col_count, figsize=(col_count*g_fw, row_count*g_fh))
        pow_axes = pow_axes.flatten()
    
    plot_state = (fig, pow_fig, axes, pow_axes, pdfs, pow_mats, g_res)
   
    return fig, pow_fig, plot_state


def fixed_plot_part(plot_state, pidx):
    if plot_state is None:
        return None, None

    fig, pow_fig, axes, pow_axes, pdfs, pow_mats, res = plot_state
    step = int(len(pdfs)/3) + 1
    
    roi_pdfs = pdfs[pidx*step: (pidx+1)*step]
    for pdf_info in roi_pdfs:
        x, x_pdf, pz_pdf, pow_idx, ax_idx = pdf_info
        axis_pdf(axes[ax_idx], x, x_pdf, label="converging pdf", color="blue")
        title = r"the %dth iterate" % pow_idx
        axis_pdf(axes[ax_idx], x, pz_pdf, title=title, label="transform result", color="green")

        div = jensenshannon(pz_pdf*res, x_pdf*res)
        axes[ax_idx].plot([], [], label="div=%0.3f"%div, color="green")
        axes[ax_idx].legend(handlelength=1.2)
        
    # fig.tight_layout()
    
    if pow_axes is None:
        return fig, None
    
    roi_pow_mats = pow_mats[pidx*step: (pidx+1)*step]
    for pow_info in roi_pow_mats:
        pow_mat, pow_idx, ax_idx = pow_info
        axis_2d_pdf(pow_axes[ax_idx], x, x, pow_mat, title="power(mat,%d)"%(pow_idx), xlabel="z", ylabel="x")
    # pow_fig.tight_layout()
    
    return fig, pow_fig


def hijack(seed, x, x_pdf):
    if seed in [16002, 16003]:
        x, x_pdf = init_x_pdf(g_st, g_et, g_num, shape_type=2, seed=100)
        left, right = (-0.5, 0.5) if seed == 16002 else (-0.7, 0.7)
        mask = np.logical_and(x > left, x < right)
        x_pdf[mask] = 0
    
    base = 17500
    left, right = int(base+g_st*100), int(base+g_et*100)
    if seed in range(left, right):
        mu, std = g_st + (seed//10*10 - left)*0.01, (seed%10+1)*0.02
        x_pdf = norm.pdf(x, mu, std)
        
    return x_pdf



