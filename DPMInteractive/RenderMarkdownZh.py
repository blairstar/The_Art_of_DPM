
import gradio as gr
from Misc import g_css, js_head, g_latex_del
js_head += """ <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/katex@0.15.3/dist/katex.min.js" integrity="sha384-0fdwu/T/EQMsQlrHCCHoH10pkPLlKA1jL5dFyUOvB3lfeT2540/2g6YgSi2BL14p" crossorigin="anonymous"></script> """



def md_introduction_zh():
    global g_latex_del

    with gr.Accordion(label="0. Introduction", elem_classes="first_md", elem_id="introduction"):
        
        gr.Markdown(
            f"""
            扩散模型[\[1\]](#dpm)[\[2\]](#ddpm)是当前图像生成和视频生成使用的主要方式，但由于其晦涩的理论，很多工程师并不能很好地理解。本文将提供一种非常直观易懂的方式，方便读者理解把握扩散模型的原理。特别地，将以互动的形式，以一维随机随机变量的扩散模型进行举例，直观解释扩散模型的多个有趣的性质。

            扩散模型是一个概率模型。概率模型主要提供两方面的功能：计算给定样本出现的概率；采样生成新样本。扩散模型侧重于第二方面，方便采样新样本，从而实现"生成"的任务。

            扩散模型与一般的概率模型(如GMM)不同，直接建模随机变量的概率分布。扩散模型采用一种间接方式，利用“随机变量变换”的方式(如图1a)，逐步将待建模的概率分布(数据分布)转变成"标准正态分布"，同时，建模学习各个变换对应的后验概率分布(图1b-c)。有了最终的标准正态分布和各个后验概率分布，则可通过祖先采样(Ancestral Sampling)的方式，从反向逐步采样得到各个随机变量$Z_T \ldots Z_2,Z_1,X$的样本。同时也可通过贝叶斯公式和全概率公式确定初始的数据分布$q(x)$。

            可能会有这样的疑问：间接的方式需要建模学习T个后验概率分布，直接方式只需要建模学习一个概率分布，为什么要选择间接的方式呢？是这样子的：初始的数据分布可能很复杂，很难用一个概率模型直接表示；而对于间接的方式，各个后验概率分布的复杂度会简单许多，可以用简单的概率模型进行拟合。下面将会看到，当满足一些条件时，后验概率分布将非常接近高斯分布，所以可以使用简单的条件高斯模型进行建模。

            <center> <img src="file/fig1.png" width="820" style="margin-top:12px"/> </center>
            <center> Figure 1: Diffusion model schematic </center>
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_introduction_zh")
    return


def md_transform_zh():
    global g_latex_del

    with gr.Accordion(label="1. How To Transform", elem_classes="first_md", elem_id="transform"):
        
        gr.Markdown(
            r"""
            为了将初始的数据分布转换为简单的标准正态分布，扩散模型采用如下的变换方式
            \begin{align}
                Z = \sqrt{\alpha} X +  \sqrt{1-\alpha}\epsilon \qquad where \quad \alpha < 1, \quad \epsilon \sim \mathcal{N}(0, I)  \tag{1.1}
            \end{align}
            其中$X\sim q(x)$是任意的随机变量，$Z\sim q(Z)$是变换后的随机变量。

            此变换可分为两个子变换。

            第一个子变换是对随机变量$X$执行一个线性变换($\sqrt{\alpha}X$)，根据文献[\[3\]](#linear_transform)的结论，线性变换使$X$的概率分布“变窄变高”，并且"变窄变高"的程度与$\alpha$的值成正比。具体可看<a href="#demo_1">Demo 1</a>，左1图为随机生成的一维的概率分布，左2图是经过线性变换后的概率分布，可以看出，与左1图相比，左2图的曲线“变窄变高”了。读者可亲自测试不同的$\alpha$值，获得更直观的理解。

            第二个子变换是“加上独立的随机噪声”($\sqrt{1-\alpha}\epsilon$)，根据文献[\[4\]](#sum_conv)的结论，“加上独立的随机变量”等效于对两个概率分布执行卷积，由于随机噪声的概率分布为高斯形状，所以相当于执行”高斯模糊“的操作。经过模糊后，原来的概率分布将变得更加平滑，与标准正态分布将更加相似。模糊的程度与噪声大小($1-\alpha$)正相关。具体可看<a href="#demo_1">Demo 1</a>，左1图是随机生成的一维概率分布，左3图是经过变换后的结果，可以看出，变换后的曲线变光滑了，棱角变少了。读者可测试不同的$\alpha$值，感受噪声大小对概率分布曲线形状的影响。左4图是综合两个子变换后的结果。
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_transform_zh")
    return


def md_likelihood_zh():
    global g_latex_del

    with gr.Accordion(label="2. Likelihood of The Transform", elem_classes="first_md", elem_id="likelihood"):

        gr.Markdown(
            r"""
            由变换的方式(式1.1)可以看出，前向条件概率$q(z|x)$的概率分布为高斯分布，且只与$\alpha$的值有关，与$q(x)$的概率分布无关。
            \begin{align}
                q(z|x) &= \mathcal{N}(\sqrt{\alpha}x,\ 1-\alpha)    \tag{2.1}
            \end{align}
            具体可看<a href="#demo_2">Demo 2</a>，左3图展示了$q(z|x)$的形状，从图中可以看到一条均匀的斜线，这意味着$q(z|x)$的均值与x线性相关，方差固定不变。$\alpha$值的大小将决定斜线宽度和倾斜程度。
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_likelihood_zh")
    return


def md_posterior_zh():
    global g_latex_del

    with gr.Accordion(label="3. Posterior of The Transform", elem_classes="first_md", elem_id="posterior"):

        gr.Markdown(
            r"""
            后验概率分布没有闭合的形式，但可以通过一些方法，推断其大概的形状，并分析影响其形状的因素。

            根据Bayes公式，有
            \begin{align}
                q(x|z) = \frac{q(z|x)q(x)}{q(z)}    \tag{3.1}
            \end{align}

            当$z$是取固定值时，$q(z)$是常数，所以$q(x|z)$的形状只与${q(z|x)q(x)}$有关。
            \begin{align}
                q(x|z)  \propto q(z|x)q(x) 	\qquad where\ z\ is\ fixed  \tag{3.2}
            \end{align}
            由式2.1可知，$q(z|x)$为高斯分布，于是有
            \begin{align}
                q(x|z)  &\propto \frac{1}{\sqrt{2\pi(1-\alpha)}}\exp{\frac{-(z-\sqrt{\alpha}x)^2}{2(1-\alpha)}}\ q(x)& 	\qquad &where\ z\ is\ fixed      \tag{3.3}   \newline
                        &=	\frac{1}{\sqrt{\alpha}} \underbrace{\frac{1}{\sqrt{2\pi}\sigma}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}}_{\text{GaussFun}}\ q(x)& \qquad &where\ \mu=\frac{z}{\sqrt{\alpha}}\quad \sigma=\sqrt{\frac{1-\alpha}{\alpha}}   \tag{3.4}
            \end{align}

            可以看出，<b>GaussFun</b>部分是关于$x$的高斯函数，均值为$\frac{z}{\sqrt{\alpha}}$，方差为$\sqrt{\frac{1-\alpha}{\alpha}}$，所以$q(x|z)$的形状由“<b>GaussFun与$q(x)$相乘</b>”决定。

            根据”乘法“的特点，可以总结$q(x|z)$函数形状具有的特点。
            <ul>
                <li> 当高斯函数的方差较小(较小噪声)，或者$q(x)$变化缓慢时，$q(x|z)$的形状将近似于高斯函数，函数形式较简单，方便建模学习。</li>
                <li> 当高斯函数的方差较大(较大噪声)，或者$q(x)$剧烈变化时，$q(x|z)$的形状将较复杂，与高斯函数有较大的差别，难以建模学习。</li>
            </ul>
            
            具体可看<a href="#demo_2">Demo 2</a>，左4图给出后验概率分布$q(x|z)$的形态，可以看出，其形状较不规则，像一条弯曲且不均匀的曲线。当$\alpha$较大时(噪声较小)，曲线将趋向于均匀且笔直。读者可调整不同的$\alpha$值，观察后验概率分布与噪声大小的关系；左5图，$\textcolor{blue}{蓝色虚线}$给出$q(x)$，$\textcolor{green}{绿色虚线}$给出式3.4中的GaussFun，$\textcolor{orange}{黄色实线}$给出两者相乘并归一化的结果，即固定z条件下后验概率$q(x|z=fixed)$。读者可调整不同z值，观察$q(x)$的波动变化对后验概率$q(x|z)$形态的影响。
            
            两个特殊状态下的后验概率分布$q(x|z)$值得考虑一下。
            <ul>
                <li> 当$\alpha \to 0$时，GaussFun的方差趋向于<b>无穷大</b>，不同$z$值的$q(x|z)$几乎变成一致，并与$q(x)$几乎相同。读者可在<a href="#demo_2">Demo 2</a>中，将$\alpha$设置为0.01，观察具体的结果。</li>
                <li> 当$\alpha \to 1$时，GaussFun的方差趋向于<b>无穷小</b>，不同$z$值的$q(x|z)$收缩成一系列不同偏移量的Dirac delta函数, 偏移量等于$z$。但有一些例外，当q(x)存在为零的区域时，其对应的q(x|z)将不再为Dirac delta函数，而是零函数。可在<a href="#demo_2">Demo 2</a>中，将$\alpha$设置为0.999，观察具体的结果。</li>
            </ul>
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_posterior_zh")
    return


def md_forward_process_zh():
    global g_latex_del

    title = "4. Transform Data Distribution To Normal Distribution"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="forward_process"):

        gr.Markdown(
            r"""
            对于任意的数据分布$q(x)$，均可连续应用上述的变换(如式4.1~4.4)，随着变换的次数的增多，输出的概率分布将变得越来越接近于标准正态分布。对于较复杂的数据分布，需要较多的次数或者较大的噪声。

            具体可看<a href="#demo_3_1">Demo 3.1</a>，第一子图是随机生成的一维概率分布，经过7次的变换后，最终的概率分布与标准正态分布非常相似。相似的程度与迭代的次数和噪声大小正相关。对于相同的相似程度，如果每次所加的噪声较大(较小的$\alpha$值)，那所需变换的次数将较少。读者可尝试不同的$\alpha$值和次数，观测最终概率分布的相似程度。

            起始概率分布的复杂度会比较高，随着变换的次数增多，概率分布$q(z_t)$的复杂度将会下降。根据第3节结论，更复杂的概率分布对应更复杂的后验概率分布，所以，为了保证后验概率分布与高斯函数较相似(较容易学习)，在起始阶段，需使用较大的$\alpha$(较小的噪声)，后期阶段可适当使用较小的$\alpha$(较大的噪声)，加快向标准正态分布转变。

            在<a href="#demo_3_1">Demo 3.1</a>的例子可以看到，随着变换次数增多，$q(z_t)$的棱角变得越来越少，同时，后验概率分布$q(z_{t-1}|z_t)$图中的斜线变得越来越笔直匀称，越来越像条件高斯分布。

            \begin{align}
                Z_1   &= \sqrt{\alpha_1} X + \sqrt{1-\alpha_1}\epsilon_1 			\tag{4.1}   \newline
                Z_2   &= \sqrt{\alpha_2} Z_1 + \sqrt{1-\alpha_2}\epsilon_2 			\tag{4.2}   \newline
                      &\dots														\notag      \newline
                Z_{t} &= \sqrt{\alpha_t}Z_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t}	    \tag{4.3}   \newline
                      &\dots													    \notag      \newline
                Z_{T} &= \sqrt{\alpha_T}Z_{T-1} + \sqrt{1-\alpha_T}\epsilon_{T}	    \tag{4.4}   \newline
                      &where \quad \alpha_t < 1   \qquad t\in {1,2,\dots,T}         \notag
            \end{align}

            把式4.1代入式4.2，同时利用高斯分布的性质，可得出$q(z_2|x)$的概率分布的形式
            \begin{align}
                z_2 &= \sqrt{\alpha_2}(\sqrt{\alpha_1}x + \sqrt{1-\alpha_1}\epsilon_1) + \sqrt{1-\alpha_2}\epsilon_2	    \tag{4.5}   \newline
                    &= \sqrt{\alpha_2\alpha_1}x + \sqrt{\alpha_2-\alpha_2\alpha_1}\epsilon_1 + \sqrt{1-\alpha_2}\epsilon_2  \tag{4.6}   \newline
                    &= \mathcal{N}(\sqrt{\alpha_1\alpha_2}x,\ 1-\alpha_1\alpha_2)                                           \tag{4.7}
            \end{align}
            
            同理，可递推得出
            \begin{align}
                q(z_t|x) &= \mathcal{N}(\sqrt{\alpha_1\alpha_2\cdots\alpha_t}x,\ 1-\alpha_1\alpha_2\cdots\alpha_t) = \mathcal{N}(\sqrt{\bar{\alpha_t}}x,\ 1-\bar{\alpha_t})  \qquad where\ \bar{\alpha_t} \triangleq \prod_{j=1}^t\alpha_j      \tag{4.8}
            \end{align}
            
            比较式4.8和式2.1的形式，可发现，两者的形式是完全一致的。如果只关注最终变换后的分布$q(z_t)$，那么连续t次的小变换可用一次大变换替代，大变换的$\alpha$是各个小变换的$\alpha$累积。
            
            在DDPM[\[2\]](#ddpm)论文中，作者使用了1000步(T=1000)，将数据分布$q(x)$转换至$q(z_T)$，$q(z_T|x)$的概率分布如下：
            \begin{align}
                q(z_T|x) &= \mathcal{N}(0.00635\ x,\ 0.99998)    \tag{4.9}
            \end{align}
            
            如果只考虑边际分布$q(z_T)$，也可使用一次变换代替，变换如下:
            \begin{align}
                Z_T = \sqrt{0.0000403}\ X + \sqrt{1-0.0000403}\ \epsilon = 0.00635\ X + 0.99998\ \epsilon 			 \tag{4.10}
            \end{align}
            可以看出，应用两种变换后，变换后的分布$q(z_T|x)$相同，因此，$q(z_T)$也相同。
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_forward_process_zh")
    return


def md_backward_process_zh():
    global g_latex_del

    title = "5. Restore Data Distribution From Normal Distribution"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="backward_process"):

        gr.Markdown(
            r"""
            如果知道了最终的概率分布$q(z_T)$及各个转换过程的后验概率$q(x|z),q(z_{t-1}|z_t)$，则可通过“贝叶斯公式”和“全概率公式”恢复数据分布$q(x)$，见式5.1~5.4。当最终的概率分布$q(z_T)$与标准正态分布很相似时，可用标准正态分布代替。
            
            具体可看<a href="#demo_3_2">Demo 3.2</a>。示例中$q(z_T)$使用$\mathcal{N}(0,1)$代替，同时通过JS Div给出了误差大小。恢复的概率分布$q(z_t)$及$q(x)$使用$\textcolor{green}{绿色曲线}$标识，原始的概率分布使用$\textcolor{blue}{蓝色曲线}$标识。可以看出，数据分布$q(x)$能够被很好地恢复回来，并且误差(JS Divergence)会小于标准正态分布替换$q(z_T)$引起的误差。
            \begin{align}
                q(z_{T-1}) &= \int q(z_{T-1},z_T)dz_T = \int q(z_{T-1}|z_T)q(z_T)dz_T               	    \tag{5.1}   \newline
                           & \dots	                                                                        \notag      \newline
                q(z_{t-1}) &= \int q(z_{t-1},z_t)dz_t = \int q(z_{t-1}|z_t)q(z_t)dz_t                       \tag{5.2}   \newline
                           & \dots	                                                                        \notag      \newline
                q(z_1)     &= \int q(z_1,z_2) dz_1    = \int q(z_1|z_2)q(z_2)dz_2                           \tag{5.3}   \newline
                q(x)       &= \int q(x,z_1) dz_1      = \int q(x|z_1)q(z_1)dz_1                             \tag{5.4}   \newline
            \end{align}
            在本文中，将上述恢复过程(式5.1~5.4)所使用的变换称之为“后验概率变换”。例如，在式5.4中，变换的输入为概率分布函数$q(z_1)$，输出为概率分布函数$q(x)$，整个变换由后验概率分布$q(x|z_1)$决定。此变换也可看作为一组基函数的线性加权和，基函数为不同条件下的$q(x|z_1)$，各个基函数的权重为$q(z_1)$。在<a href="#posterior_transform">第7节</a>，将会进一步介绍此变换的一些有趣性质。
            
            在<a href="#posterior">第3节</a>中，我们考虑了两个特殊的后验概率分布。接下来，分析其对应的”后验概率变换“。
            <ul>
                <li> 当$\alpha \to 0$时，不同$z$值的$q(x|z)$均与$q(x)$几乎相同，也就是说，线性加权和的基函数几乎相同。此状态下，不管输入如何变化，变换的输出总为$q(x)$。</li>
                <li> 当$\alpha \to 1$时，不同$z$值的$q(x|z)$收缩成一系列不同偏移量的Dirac delta函数及零函数。此状态下，只要输入分布的支撑集(support set)包含于$q(x)$的支撑集，变换的输出与输入将保持一致。</li>
            </ul>
            
            在<a href="#forward_process">第4节</a>中提到，DDPM[\[2\]](#ddpm)论文所使用的1000次变换可使用一次变换表示：
            \begin{align}
                Z_T = \sqrt{0.0000403}\ X + \sqrt{1-0.0000403}\ \epsilon = 0.00635\ X + 0.99998\ \epsilon 			 \tag{5.5}
            \end{align}
            由于$\alpha=0.0000403$非常小，其对应的GaussFun(式3.4)的标准差达到157.52，而$X$的范围限制在$[-1, 1]$，远小于GaussFun的标准差。在$x \in [-1, 1]$范围内，GaussFun应该接近于常量，没有什么变化，所以不同的$z_T$对应的$q(x|z_T)$均与$q(x)$几乎相同。在这种状态下，对于$q(x|z_T)$相应的后验概率变换，不管输入分布是什么，输出分布都将是$q(x)$。
            
            <b>所以，理论上，在DDPM模型中，无需非得使用标准正态分布代替$q(z_T)$，也可使用其它任意的分布代替。</b>
            
            读者可亲自做一个类似的实验。在<a href="#demo_3_1">Demo 3.1</a>中，将start_alpha设置0.25，end_alpha也设置为0.25，step设置为7，此时$q(z_7)=\sqrt{0.000061}X + \sqrt{1-0.000061}\epsilon$，与DDPM的$q(z_T)$基本相似。点击<b>apply</b>执行前向变换($\textcolor{blue}{蓝色曲线}$)，为接下来的反向恢复做准备。在<a href="#demo_3_2">Demo 3.2</a>中，noise_ratio设置为1，为末端分布$q(z_7)$引入100%的噪声，切换nose_random_seed的值可改变噪声的分布，取消选择backward_pdf，减少画面的干扰。点击<b>apply</b>将通过后验概率变换恢复$q(x)$，将会看到，不管输入的$q(z_7)$的形状如何，恢复的$q(x)$均与原始的$q(x)$完全相同, JS Divergence为0，恢复的过程使用$\textcolor{red}{红色曲线}$画出。
             
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_backward_process_zh")
    return


def md_fit_posterior_zh():
    global g_latex_del

    title = "6. Fitting Posterior With Conditional Gaussian Model"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="fit_posterior"):
        
        # because of the render bug in gradio markdown, some formulas are render in ExtraBlock.js
        gr.Markdown(
            r"""
            由<a href="#posterior">第3节</a>前半部分可知，各个后验概率分布是未知的，并且与$q(x)$有关。所以，为了恢复数据分布或者从数据分布中采样，需要对各个后验概率分布进行学习估计。

            由<a href="#posterior">第3节</a>后半部分可知，当满足一定条件时，各个后验概率分布$q(x|z)、q(z_{t-1}|z_t)$近似于高斯概率分布，所以可通过构建一批条件高斯概率模型$p(x|z),p(z_{t-1}|z_t)$，学习拟合对应的$q(x|z),q(z_{t-1}|z_t)$。

            由于模型表示能力和学习能力的局限性，拟合过程会存在一定的误差，进一步会影响恢复$q(x)$的准确性。拟合误差大小与后验概率分布的形状有关。由<a href="#posterior">第3节</a>可知，当$q(x)$较复杂或者所加噪声较大时，后验概率分布会较复杂，与高斯分布差别较大，从而导致拟合误差，进一步影响恢复$q(x)$。
            
            具体可看<a href="#demo_3_3">Demo 3.3</a>，读者可测试不同复杂程度的$q(x)$和$\alpha$，观看后验概率分布$q(z_{t-1}|z_t)$的拟合程度，以及恢复$q(x)$的准确度。恢复的概率分布使用$\textcolor{orange}{橙色}$标识，同时也通过JS divergence给出误差。

            关于拟合的目标函数，与其它概率模型类似，可$\textcolor{red}{优化交叉熵损失}$，使$p(z_{t-1}|z_t)$逼近于$q(z_{t-1}|z_t)$。由于$(z_{t-1}|z_t)$是条件概率，所以需要综合考虑各个条件，以<b>各个条件发生的概率$q(z_t)$</b>加权平均<b>各个条件对应的交叉熵</b>。最终的损失函数形式如下:
            \begin{align}
                loss &= -\int q(z_t)\ \overbrace{\int q(z_{t-1}|z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}}^{\text{Cross Entropy}}\ dz_t        \tag{6.1}     \newline
                     &= -\iint q(z_{t-1},z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                         \tag{6.2} 
            \end{align}
            也可以KL散度作为目标函数进行优化，KL散度与交叉熵是等价的[\[10\]](#ce_kl)。
            <span id="zh_fit_0">
                loss &= \int q(z_t) KL(q(z_{t-1}|z_t) \Vert \textcolor{blue}{p(z_{t-1}|z_t)})dz_t                                                             \tag{6.3}      \newline
                     &= \int q(z_t) \int q(z_{t-1}|z_t) \log \frac{q(z_{t-1}|z_t)}{\textcolor{blue}{p(z_{t-1}|z_t)}} dz_{t-1} dz_t                                 \tag{6.4}      \newline 
                     &= -\int q(z_t)\ \underbrace{\int q(z_{t-1}|z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}}{underline}{\text{Cross Entropy}}\ dz_t + \underbrace{\int q(z_t) \int q(z_{t-1}|z_t) \log q(z_{t-1}|z_t)}{underline}{\text{Is Constant}} dz  \tag{6.5}
            </span>
            
            式6.2的积分没有闭合的形式，不能直接优化。可使用蒙特卡罗(Monte Carlo)积分近似计算，新的目标函数如下:
            \begin{align}
                loss &= -\iint q(z_{t-1},z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                     \tag{6.6}      \newline
                     &\approx -\sum_{i=0}^N \log \textcolor{blue}{p(Z_{t-1}^i|Z_t^i)} \qquad where \quad (Z_{t-1}^i,Z_t^i) \sim q(z_{t-1},z_t)      \tag{6.7} 
            \end{align}

            上述的样本$(Z_{t-1}^i,Z_t^i)$服从联合概率分布$q(z_{t-1},z_t)$，可通过祖先采样的方式采样得到。具体方式如下：通过正向转换的方式(式4.1~4.4)，逐步采样$X,Z_1,Z_2\dots Z_{t-1},Z_t$，然后留下$(Z_{t-1},Z_t)$作为一个样本。但这种采样方式比较慢，可利用$q(z_t|x)$概率分布已知的特点(式4.8)加速采样，先从$q(x)$采样$X$，然后由$q(z_{t-1}|x)$采样$Z_{t-1}$，最后由$q(z_t|z_{t-1})$采样$Z_t$，于是得到一个样本$(Z_{t-1},Z_t)$。

            可能有些人会有疑问，式6.3的形式跟DPM[\[1\]](#dpm)和DDPM[\[2\]](#ddpm)论文里的形式不太一样。实际上，这两个目标函数是等价的，下面给出证明。

            对于一致项(Consistent Term)，证明如下:

            \begin{align}
                loss &= -\iint q(z_{t-1},z_t)\ \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                                                            \tag{6.8}           \newline
                     &= -\iint \int q(x)q(z_{t-1}, z_t|x)dx\ \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                                              \tag{6.9}           \newline
                     &= \overbrace{\iint \int q(x)q(z_{t-1}, z_t|x) \log q(z_{t-1}|z_t,x)dxdz_{t-1}dz_t}^{\text{This Term Is Constant And Is Denoted As}\ \textcolor{orange}{C_1}}          \tag{6.10}          \newline
                     &\quad - \iint \int q(x)q(z_{t-1}, z_t|x) \log \textcolor{blue}{p(z_{t-1}|z_t)}dxdz_{t-1}dz_t - \textcolor{orange}{C_1}                                                \tag{6.11}          \newline
                     &= \iint \int q(x)q(z_{t-1},z_t|x) \log \frac{q(z_{t-1}|z_t,x)}{\textcolor{blue}{p(z_{t-1}|z_t)}}dxdz_{t-1}dz_t - \textcolor{orange}{C_1}                              \tag{6.12}          \newline
                     &= \iint q(x)q(z_t|x)\int q(z_{t-1}|z_t,x) \log \frac{q(z_{t-1}|z_t,x)}{\textcolor{blue}{p(z_{t-1}|z_t)}}dz_{t-1}\ dz_tdx - \textcolor{orange}{C_1}                   \tag{6.13}          \newline
                     &= \iint \ q(x)q(z_t|x) KL(q(z_{t-1}|z_t,x) \Vert \textcolor{blue}{p(z_{t-1}|z_t)}) dz_t dx - \textcolor{orange}{C_1}                                                  \tag{6.14}          \newline
                     &\propto \iint \ q(x)q(z_t|x) KL(q(z_{t-1}|z_t,x) \Vert \textcolor{blue}{p(z_{t-1}|z_t)}) dz_t dx                                                                      \tag{6.15}          \newline
            \end{align}

            上式中的$C_1$项是一个固定值，不包含待优化的参数，其中，$q(x)$是固定的概率分布，$q(z_{t-1}|z_t)$也是固定概率分布，具体形式由$q(x)$及系数$\alpha$确定。

            对于重构项(Reconstruction Term)，可通过类似的方式证明：
            \begin{align}
                loss &= -\int q(z_1)\overbrace{\int q(x|z_1)\log \textcolor{blue}{p(x|z_1)}dx}^{\text{Cross Entropy}}\ dz_1     \tag{6.16}   \newline
                &= -\iint q(z_1,x)\log \textcolor{blue}{p(x|z_1)}dxdz_1                 \tag{6.17}   \newline
                &= -\int q(x)\int q(z_1|x)\log \textcolor{blue}{p(x|z_1)}dz_1\ dx       \tag{6.18}
            \end{align}
            
            因此，式6.1的目标函数与DPM的目标函数是等价的。
             
            根据一致项证明的结论，以及交叉熵与KL散度的关系，可得出一个有趣的结论：
            <span id="zh_fit_1">
                \mathop{\min}{underline}{\textcolor{blue}{p}} \int q(z_t) KL(q(z_{t-1}|z_t) \Vert \textcolor{blue}{p(z_{t-1}|z_t)})dz_t  \iff  \mathop{\min}{underline}{\textcolor{blue}{p}} \iint \ q(z_t)q(x|z_t) KL(q(z_{t-1}|z_t,x) \Vert \textcolor{blue}{p(z_{t-1}|z_t)})dxdz_t           \tag{6.19}
            </span>
            比较左右两边的式子，可以看出，右边的目标函数比左边的目标函数多了一个条件变量$X$，同时也多了一个关于$X$积分，并且以$X$的发生的概率$q(x|z_t)$作为积分的加权系数。
            
            依照类似的思路，可推导出一个更通用的关系：
            <span id="zh_fit_2">
                \mathop{\min}{underline}{\textcolor{blue}{p}}  KL(q(z) \Vert \textcolor{blue}{p(z)})  \iff  \mathop{\min}_{\textcolor{blue}{p}} \int \ q(x) KL(q(z|x) \Vert \textcolor{blue}{p(z)})dx               \tag{6.20}
            </span>
            关于此结论的详细推导，可见<a href="#cond_kl">Appendix A</a>。
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_fit_posterior_zh")
    return


def md_posterior_transform_zh():
    global g_latex_del

    with gr.Accordion(label="7. Posterior Transform", elem_classes="first_md", elem_id="posterior_transform"):

        gr.Markdown(
            r"""
            <h3 style="font-size:18px"> 压缩映射及收敛点 </h3>
            \begin{align}
                q(x) &= \int q(x,z) dz = \int q(x|z)q(z)dz      \tag{7.1}
            \end{align}

            经过大量一维随机变量的实验发现，后验概率变换呈现出“压缩映射”(Contraction Mapping[\[6\]](#ctr))的特征。也是说，对任意的两个概率分布$q_{i1}(z)和q_{i2}(z)$，经过后验概率变换后得到$q_{o1}(x)$和$q_{o2}(x)$，$q_{o1}(z)$和$q_{o2}(z)$的距离<b>总是小于</b>$q_{i1}(x)$和$q_{i2}(x)$的距离。这里的距离可使用JS Divergence或Total Variance或度量。并且，这个压缩映射的压缩程度跟所加噪声大小正相关。
            \begin{align}
                dist(q_{o1}(z),\ q_{o2}(z)) < dist(q_{i1}(x),\ q_{i2}(x))                   \tag{7.2}
            \end{align}

            读者可查看<a href="#demo_4_1">Demo 4.1</a>，左侧三个图呈现一个变换的过程，左1图是任意的数据分布$q(x)$，左3图是变换后的概率分布，左2图是后验概率分布。可更改随机种子生成新的数据分布，调整$\alpha$值引入不同程度的噪声。左侧最后两个图展示变换的“压缩性质”，左4图展示随机生成的两个输入分布，同时给出其距离度量值$div_{in}$；左5图展示经过变换后的两个输出分布，输出分布之间的距离标识为$div_{out}$。读者可改变输入的随机种子，切换不同的输入。可在图中看到，对于任意的输入，$div_{in}$总是小于$div_{out}$。另外，也可改变$\alpha$的值，将会看到，$\alpha$越小(噪声越大)，$\frac{div_{out}}{div_{in}}$的比值也越小，即收缩率越大。
            
            由Banach fixed-point theorem<a href="#fixed_point">[5]</a>可知，压缩映射存在惟一一个定点(收敛点)。也就是说，对于任意的输入分布，可以连续迭代应用“后验概率变换”，只要迭代次数足够多，最终都会输出同一个分布。经过大量一维随机变量实验发现，定点(收敛点)<b>位于$q(x)$附近</b>。并且，与$\alpha$的值有关，$\alpha$越小(噪声越大)，离得越近。
            
            读者可看<a href="#demo_4_2">Demo 4.2</a>，此部分展示迭代收敛的例子。选择合适的迭代次数，点中“apply iteration transform”，将逐步画出迭代的过程，每个子图均会展示各自变换后的输出分布($\textcolor{green}{绿色曲线}$)，收敛的参考点分布$q(x)$以$\textcolor{blue}{蓝色曲线}$画出，同时给出输出分布与$q(x)$之间的距离$dist$。可以看出，随着迭代的次数增加，输出分布与$q(x)$越来越相似，并最终会稳定在$q(x)$附近。对于较复杂的分布，可能需要较多迭代的次数或者较大的噪声。迭代次数可以设置为上万步，但会花费较长时间。

            对于一维离散的情况，$q(x|z)$将离散成一个矩阵(记为$Q_{x|z}$)，$q(z)$离散成一个向量(记为$\boldsymbol{q_i}$)，积分操作$\int q(x|z)q(z)dz$将离散成"矩阵-向量"乘法操作，所以后验概率变换可写成
            \begin{align}
                \boldsymbol{q_o} &= Q_{x|z}\ \boldsymbol{q_i} &             \quad\quad        &\text{1 iteration}         \tag{7.3}       \newline
                \boldsymbol{q_o} &= Q_{x|z}\ Q_{x|z}\ \boldsymbol{q_i} &    \quad\quad        &\text{2 iteration}         \tag{7.4}       \newline
                    & \dots &                                                                                             \notag          \newline
                \boldsymbol{q_o} &= (Q_{x|z})^n\ \boldsymbol{q_i} &         \quad\quad        &\text{n iteration}         \tag{7.5}       \newline
            \end{align}
            于是，为了更深入地理解变换的特点，<a href="#demo_4_2">Demo 4.2</a>也画出矩阵$(Q_{x|z})^n$的结果。从图里可以看到，当迭代趋向收敛时，矩阵$(Q_{x|z})^n$的行向量将变成一个常数向量，即向量的各分量都相等。在二维密度图里将表现为一条横线。

            在<a href="#proof_ctr">Appendix B</a>中，将会提供一个证明，当$q(x)$和$\alpha$满足一些条件时，后验概率变换是一个严格的压缩映射。

            关于定点分布与输入分布q(x)之间距离的关系，目前尚不能严格证明。
            
            <h3 style="font-size:18px"> 恢复数据分布过程中的抗噪声能力 </h3>
            由上面的分析可知，当满足一些条件时，"后验概率变换"是一个压缩映射，所以存在如下的关系：
            \begin{align}
                dist(q(x),\ q_o(x)) < dist(q(z),\ q_i(z))         \tag{7.12}
            \end{align}
            其中，$q(z)$是理想的输入分布，$q(x)$理想的输出分布，$q_i(x)$是任意的输入分布，$q_o(x)$是$q_i(z)$经过变换后的输出分布。
            
            上式表明，输出的分布$q_o(x)$与理想输出分布q(x)之间的距离总会</em>小于</em>输入分布$q_i(z)$与理想输入分布q(x)的距离。于是，"后验概率变换"具备一定的抵抗噪声能力。这意味着，在恢复$q(x)$的过程中(<a href="#backward_process">第5节</a>)，哪怕输入的“末尾分布$q(z_T)”$存在一定的误差，经过一系列变换后，输出的“数据分布$q(x)$“的误差也会比输入的误差更小。
            
            具体可看<a href="#demo_3_2">Demo 3.2</a>，通过增加“noise ratio”的值可以向“末尾分布$q(z_T)$”添加噪声，点击“apply”按钮将逐步画出恢复的过程，恢复的分布以$\textcolor{red}{红色曲线}$画出，同时也会通过JS散度标出误差的大小。将会看到，恢复的$q(x)$的误差总是小于$q(z_T)$的误差。
            
            由上面的讨论可知，$\alpha$越小(即变换过程中使用的噪声越大)，压缩映射的压缩率越大，于是，抗噪声的能力也越强。
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_posterior_transform_zh")
    return


def md_deconvolution_zh():
    global g_latex_del
    
    title = "8. Can the data distribution be restored by deconvolution?"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="deconvolution"):

        gr.Markdown(
            r"""
            在<a href="#introduction">第1节</a>中提到，式2.1的变换可分为两个子变换，第一个子变换为”线性变换“，第二个为“加上独立高斯噪声”。线性变换相当于对概率分布进行拉伸变换，所以存在逆变换。"加上独立高斯噪声”相当于对概率分布执行卷积操作，卷积操作可通过逆卷积恢复。所以，理论上，可通过“逆线性变换”和“逆卷积”从最终的概率分布$q(z_T)$恢复数据分布$q(x)$。
            
            但实际上，会存在一些问题。由于逆卷积对误差极为敏感，具有很高的输入灵敏度，很小的输入噪声就会引起输出极大的变化[\[11\]](#deconv_1)[\[12\]](#deconv_2)。而在扩散模型中，会使用标准正态分布近似代替$q(z_T)$，因此，在恢复的起始阶段就会引入噪声。虽然噪声较小，但由于逆卷积的敏感性，噪声会逐步放大，影响恢复。
            
            另外，也可以从另一个角度理解“逆卷积恢复”的不可行性。由于前向变换的过程(式4.1~4.4)是确定的，所以卷积核是固定的，因此，相应的“逆卷积变换“也是固定的。由于起始的数据分布$q(x)$可以是任意的分布，所以，通过一系列固定的“卷积正变换”，可以将任意的概率分布转换成近似$\mathcal{N}(0,I)$的分布。如“逆卷积变换“可行，则意味着，可用一个固定的“逆卷积变换"，将$\mathcal{N}(0,I)$分布恢复成任意的数据分布$q(x)$，这明显是一个悖论。同一个输入，同一个变换，不可能会有多个输出。
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_deconvolution_zh")
    return


def md_cond_kl_zh():
    global g_latex_del

    title = "Appendix A Conditional KL Divergence"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="cond_kl"):
        gr.Markdown(
            r"""
            本节主要介绍<b>KL散度</b>与<b>条件KL散度</b>之间的关系。在正式介绍之前，先简单介绍<b>熵</b>和<b>条件熵</b>的定义，以及两者之间存在的不等式关系，为后面的证明作准备。

            <h3 style="font-size:18px">熵及条件熵</h3>
            对于任意两个随机变量$Z,X$，<b>熵</b>(Entropy)定义如下<a href="#entropy">[16]</a>：
            \begin{align}
               \mathbf{H}(Z) = \int -p(z)\log{p(z)}dz    \tag{A.1}
            \end{align}
            <b>条件熵</b>(Conditional Entropy)的定义如下<a href="#cond_entropy">[17]</a>：
            \begin{align}
               \mathbf{H}(Z|X) = \int p(x) \overbrace{\int -p(z|x)\log{p(z|x)}dz}^{\text{Entropy}}\ dx    \tag{A.2}
            \end{align}
            两者存在如下的不等式关系：
            \begin{align}
               \mathbf{H}(Z|X) \le \mathbf{H}(Z)         \tag{A.3}
            \end{align}
            也就是说，<b>条件熵总是小于或者等于熵</b>，当且仅当X与Z相互独立时，两者相等。此关系的证明可看文献<a href="#cond_entropy">[17]</a>。
            
            <h3 style="font-size:18px"> KL散度及条件KL散度 </h3>
            仿照条件熵定义的方式，引入一个新定义，<b>条件KL散度</b>，记为$KL_{\mathcal{C}}$。由于KL散度的定义是非对称的，所以存在两种形式，如下：
            \begin{align}
               KL_{\mathcal{C}}(q(z|x) \Vert \textcolor{blue}{p(z)}) = \int \ q(x) KL(q(z|x) \Vert \textcolor{blue}{p(z)})dx                        \tag{A.4}     \newline
               KL_{\mathcal{C}}(q(z) \Vert \textcolor{blue}{p(z|x)}) = \int \ \textcolor{blue}{p(x)} KL(q(z) \Vert \textcolor{blue}{p(z|x)})dx      \tag{A.5}
            \end{align}
            
            与条件熵类似，<b>两种形式的条件KL散度</b>也都存在类似的不等式关系：
            \begin{align}
               KL_{\mathcal{C}}(q(z|x) \Vert \textcolor{blue}{p(z)}) \ge KL(q(z) \Vert \textcolor{blue}{p(z)})        \tag{A.6}   \newline
               KL_{\mathcal{C}}(q(z) \Vert \textcolor{blue}{p(z|x)}) \ge KL(q(z) \Vert \textcolor{blue}{p(z)})        \tag{A.7}
            \end{align}
            也就是说，<b>条件KL散度总是大于或者等于KL散度</b>，当且仅当X与Z相互独立时，两者相等。
             
            下面对式A.5和式A.6的结论分别证明。
            
            对于式A.6，证明如下：
            \begin{align}
                KL_{\mathcal{C}}(q(z|x) \Vert \textcolor{blue}{p(z)}) &= \int \ q(x) KL(q(z|x) \Vert \textcolor{blue}{p(z)})dx      \tag{A.8}    \newline
                    &= \iint q(x) q(z|x) \log \frac{q(z|x)}{\textcolor{blue}{p(z)}}dzdx                                             \tag{A.9}    \newline
                    &= -\overbrace{\iint - q(x)q(z|x) \log q(z|x) dzdx}^{\text{Conditional Entropy }\mathbf{H}_q(Z|X)} - \iint q(x) q(z|x) \log \textcolor{blue}{p(z)} dzdx          \tag{A.10}   \newline
                    &= -\mathbf{H}_q(Z|X) - \int \left\lbrace \int q(x) q(z|x)dx \right\rbrace \log \textcolor{blue}{p(z)}dz                                                        \tag{A.11}   \newline
                    &= -\mathbf{H}_q(Z|X) + \overbrace{\int - q(z) \log p(z)dz}^{\text{Cross Entropy}}                                                                              \tag{A.12}   \newline
                    &= -\mathbf{H}_q(Z|X) + \int q(z)\left\lbrace \log\frac{q(z)}{\textcolor{blue}{p(z)}} -\log q(z)\right\rbrace dz                                                \tag{A.13}   \newline
                    &= -\mathbf{H}_q(Z|X) + \int q(z)\log\frac{q(z)}{\textcolor{blue}{p(z)}}dz + \overbrace{\int - q(z)\log q(z)dz}^{\text{Entropy } \mathbf{H}_q(Z)}               \tag{A.14}   \newline
                    &= KL(q(z) \Vert \textcolor{blue}{p(z)}) + \overbrace{\mathbf{H}_q(Z) - \mathbf{H}_q(Z|X)}^{\ge 0}              \tag{A.15}  \newline
                    &\le KL(q(z) \Vert \textcolor{blue}{p(z)})      \tag{A.16}
            \end{align}
            其中式A.15应用了"条件熵总是小于或者等于熵"的结论。于是，得到式A.6的关系。
            
            对于式A.7，证明如下：
            \begin{align}
                KL(\textcolor{blue}{q(z)} \Vert p(z)) &= \int \textcolor{blue}{q(z)}\log\frac{\textcolor{blue}{q(z)}}{p(z)}dz    \tag{A.15}          \newline
                        &= \int q(z)\log\frac{q(z)}{\int p(z|x)p(x)dx}dz 						 	\tag{A.16}          \newline
                        &= \textcolor{orange}{\int p(x)dx}\int q(z)\log q(z)dz - \int q(z)\textcolor{red}{\log\int p(z|x)p(x)dx}dz	\qquad \ \textcolor{orange}{\int p(x)dx=1}			    \tag{A.17}      \newline
                        &\le \iint p(x) q(z)\log q(z)dzdx - \int q(z)\textcolor{red}{\int p(x)\log p(z|x)dx}dz \ \qquad 	 \textcolor{red}{\text{jensen\ inequality}}                     \tag{A.18}      \newline
                        &= \iint p(x)q(z)\log q(z)dzdx - \iint p(x)q(z)\log p(z|x)dzdx			 	    \tag{A.19}          \newline
                        &= \iint p(x)q(z)(\log q(z) - \log p(z|x))dzdx								    \tag{A.20}          \newline
                        &= \iint p(x)q(z)\log \frac{q(z)}{p(z|x)}dzdx								    \tag{A.21}          \newline
                        &= \int p(x)\left\lbrace \int q(z)\log \frac{q(z)}{p(z|x)}dz\right\rbrace dx    \tag{A.22}          \newline
                        &= \int p(x)KL(\textcolor{blue}{q(z)} \Vert p(z|x))dx                           \tag{A.23}          \newline
                        &= KL_{\mathcal{C}}(q(z) \Vert \textcolor{blue}{p(z|x)})                        \tag{A.24}
            \end{align}
            于是，得到式A.7的关系。
           
            从式A.15可得出另外一个<b>重要的结论</b>。
            
            KL散度常用于拟合数据的分布。在此场景中，数据潜在的分布用$q(z)$表示，参数化的模型分布用$\textcolor{blue}{p_\theta(z)}$表示。在优化的过程中，由于$q(z|x)$和$q(x)$均保持不变，所以式A.15中的$\mathbf{H}(Z) - \mathbf{H}(Z|X)$为一个常数项。于是，可得到如下的关系
            <span id="zh_cond_kl_2">
                \mathop{\min}{underline}{\textcolor{blue}{p_\theta}}  KL(q(z) \Vert \textcolor{blue}{p_\theta(z)})  \iff  \mathop{\min}{underline}{\textcolor{blue}{p_\theta}} \int \ q(x) KL(q(z|x) \Vert \textcolor{blue}{p_\theta(z)})dx   \tag{A.25}
            </span>
            
            把上述的关系与Denoised Score Matching<a href="#dsm">[18]</a>作比较，可发现一些相似的地方。两者均引入一个新变量$X$，并且将拟合的目标分布q(z)代替为q(z|x)。代替后，由于q(z|x)是条件概率分布，所以，两者均考虑了所有的条件，并以条件发生的概率$q(x)$作为权重系数执行加权和。
            <span id="zh_cond_kl_3">
                \mathop{\min}{underline}{\textcolor{blue}{\psi_\theta}} \frac{1}{2} \int q(z) \left\lVert \textcolor{blue}{\psi_\theta(z)} - \frac{\partial q(z)}{\partial z} \right\rVert^2 dz \iff  \mathop{\min}{underline}{\textcolor{blue}{\psi_\theta}} \int q(x)\ \overbrace{\frac{1}{2} \int q(z|x) \left\lVert \textcolor{blue}{\psi_\theta(z)} - \frac{\partial q(z|x)}{\partial z} \right\rVert^2 dz}^{\text{Score Matching of }q(z|x)}\ dx      \tag{A.26}
            </span>
            
            上述加权和的操作有点类似于"全概率公式消元"。
            \begin{align}
                q(z) = \int q(z,x) dx = \int q(x) q(z|x) dx     \tag{A.27}
            \end{align}
            
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_cond_kl_zh")
    return


def md_proof_ctr_zh():
    global g_latex_del

    title = "Appendix B Proof of Contraction"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="proof_ctr"):
        gr.Markdown(
            r"""
            <center> <img src="file/fig2.png" width="960" style="margin-top:12px"/> </center>
            <center> Figure 2: Only one component in support </center>
            
            本节将证明，当$q(x)$及$\alpha$满足一些条件时，后验概率变换是一个压缩映射，并存在惟一收敛点。
            
            下面分四种情况进行证明。证明的过程假设随机变量是离散型的，因此，后验概率变换可看作是一个<b>离散Markov Chain</b>的一步转移，后验概率$q(x|z)$对应于<b>转移矩阵</b>(Transfer Matrix)。连续型的变量可认为是无限多状态的离散型变量。
            
            <ol style="list-style-type:decimal">
            <li> 当$q(x)$均大于0时，后验概率变换矩阵$q(x|z)$将大于0，于是此矩阵是一个$\textcolor{red}{不可约}\textcolor{green}{非周期}$的Markov Chain的转移矩阵，根据文献<a href="#mc_basic_p6">[13]</a>的结论，此变换是一个关于Total Variance度量的压缩映射，于是，根据Banach fixed-point theorem，此变换存在惟一定点(收敛点)。</li>
             
            <li> 当$q(x)$部分大于0，并且$q(x)$的支撑集($q(x)$大于0的区域)只存在一个连通域时(图2)，由式(3.4)可分析出几个结论：
            
            <ol style="list-style-type:lower-alpha; padding-inline-start: 0px;font-size:16px;">
            <li> 当$z$和$x$在支撑集内时，由于$q(x)$和GaussFun均大于0，所以，转移矩阵的对角元素$\{q(x|z)|z=x\}$大于0。这意味着，支撑集内的状态是$\textcolor{green}{非周期}$的。</li>
            
            <li> 当$z$和$x$在支撑集内时，由于GaussFun的支撑集存在一定的半径，所以，在对角元素上下附近区域内的$\{q(x|z)|x=z+\epsilon\}$也大于0。这意味着，支撑集内的状态可相互访问(accessible)，形成一个$\textcolor{red}{\text{Communication Class}}$<a href="#mc_basic_d4">[14]</a>。</li>
                
            <li> 当<em>$z$在支撑集内</em>且<em>$x$在支撑集外</em>时，${q(x|z)}$全为0。这意味着，支撑集内的状态<em>不可访问</em>支撑集外的状态(图2b的inaccessible区域)。</li>
                
            <li> 当<em>$z$在支撑集外</em>且<em>$x$在支撑集内</em>时，由于GaussFun的支撑集存在一定的范围，所以，存在部分扩展区域(图2b的extension区域)，其对应的$\{q(x|z)|x\in support\}$不全为0。这意味着，此部分扩展区域的状态可<em>单向</em>访问(access)支撑集内的状态(图2b的unidirectional区域)。</li>
                
            <li> 当<em>$z$在支撑集外</em>且<em>$x$在支撑集外</em>时，对应的$q(x|z)$全为0。这意味着，支撑集外的状态不会转移至支撑集外的状态，也就是说，支撑集外的状态只来源于支撑集内的状态。</li>
            
            <p style="margin-top:8px">
            由(c)可知，支撑集内的状态<em>不会转移到</em>支撑集外的状态，由(a)和(b)可知，支撑集内的状态是非周期且构成一个Communicate Class，所以，支撑集内的状态独立构成一个不可约且非周期的Markov Chain，根据文献<a href="#mc_limit">[7]</a>中Theorem 11.4.1的结论，当$n\to+\infty$时，$q(x|z)^n$收敛于一个固定矩阵，并且矩阵每个列向量都相同。这意味着，对于不同的z，$q(x|z)^n$都相同(可见图2c)。另外，由(d)和(e)可知，存在部分支撑集外的z状态，能转移至支撑集内，并且会带着支撑集内的信息转移回支撑集外，于是，此部分z状态对应的$q(x|z)$(图2c的$q(x|z_{ex})$区域)也会等于支撑集内对应的$q(x|z)$(图2c的$q(x|z_{sup})$区域)。
            </p>
            
            <p style="margin-top:8px">
            所以，可以得出结论，当状态限制在支撑集和两个扩展区域内时，$\lim_{n\to\infty}{q(x|z)^n}$会收敛于一个固定矩阵，并且每个列向量均相同。于是，对于任意的输入分布，如果连续应用足够多后验概率变换，最终会收敛于一个固定分布，此分布等于收敛的矩阵的列向量。根据文献<a href="#fp_converse">[9]</a>的结论，当迭代变换收敛于惟一定点时，此变换是关于某个metric的Contraction Mapping。
            </p>

            </ol>
            </li>
            
            <li> 当$q(x)$部分大于0，$q(x)$的支撑集存在多个连通域，并且各个连通域的最大距离<b>能</b>被相应的GaussFun的支撑集所覆盖时，那各个连通域内的状态构成一个Communicate Class。如图3所示，$q(x)$存在两个连通域，在第一个连通域的边缘，$q(x|z=-0.3)$对应的GaussFun的支撑集能跨越间隙到达第二个连通域，于是第一个连通域的状态能<em>访问</em>第二个连通域的状态；在第二个连通域的边缘，$q(x|z=0)$对应的GaussFun的支撑集也能跨越间隙到达第一个连通域，于是第二个连通域的状态能<em>访问</em>第一个连通域的状态，所以两个连通域构成一个Communicate Class。因此，与单个连通域的情况类似，当状态限制在各个连通域、间隙及扩展区域内时，后验概率变换存在惟一一个迭代收敛点，并且是关于某个metric的压缩映射。</li>
            
            <li> 当$q(x)$部分大于0，$q(x)$的支撑集存在多个连通域时，并且各个连通域的最大距离<b>不能</b>被相应的GaussFun的支撑集所覆盖时，那各个连通域内的状态构成多个Communicate Class，如图4所示。此情况下，当$n\to\infty$时，$q(x|z)^n$也会收敛于一个固定矩阵，但每个列向量不尽相同。所以，后验概率变换不是一个严格的压缩映射。但当输入分布的状态限制在单个Communicate Class及相应的扩展范围内时，后验概率变换也是一个压缩映射，存在惟一收敛点。</li>
            </ol>
            
            <center> <img src="file/fig3.png" width="960" style="margin-top:12px"/> </center>
            <center> Figure 3: Two component which can communicate with each other </center>
            
            <center> <img src="file/fig4.png" width="960" style="margin-top:12px"/> </center>
            <center> Figure 4: Two component which <b>cannot</b> communicate with each other </center>
            
            另外，后验概率变换存在一个更通用的关系，与$q(x|z)$的具体值无关: 两个输出分布的之间的Total Variance距离总是会<b>小于等于</b>对应输入分布之间的Total Variance距离，即
            \begin{align}
                dist(q_{o1}(x),\ q_{o2}(x)) \le dist(q_{i1}(z),\ q_{i2}(z))    \tag{B.1}
            \end{align}
            下面通过离散的形式给出证明：
            \begin{align}
                      \lVert q_{o1}-q_{o2}\rVert_{TV} &= \lVert Q_{x|z}q_{i1} - Q_{x|z}q_{i2}\rVert_{TV}                                                                                                                    \tag{B.2}         \newline
                                                      &=   \sum_{m}\textcolor{red}{|}\sum_{n}Q_{x|z}(m,n)q_{i1}(n) - \sum_{n}Q_{x|z}(m,n)q_{i2}(n)\textcolor{red}{|}                                                        \tag{B.3}         \newline
                                                      &=   \sum_{m}\textcolor{red}{|}\sum_{n}Q_{x|z}(m,n)(q_{i1}(n) - q_{i2}(n))\textcolor{red}{|}                                                                          \tag{B.4}         \newline
                                                      &\leq \sum_{m}\sum_{n}Q_{x|z}(m,n)\textcolor{red}{|}(q_{i1}(n) - q_{i2}(n))\textcolor{red}{|}             \qquad \qquad \qquad \text{Absolute value inequality}       \tag{B.5}         \newline
                                                      &=   \sum_{n}\textcolor{red}{|}(q_{i1}(n) - q_{i2}(n))\textcolor{red}{|} \sum_{m} Q_{x|z}(m,n)            \qquad \qquad \qquad \sum_{m} Q_{x|z}(m,n) = 1              \tag{B.6}         \newline
                                                      &=   \sum_{n}\textcolor{red}{|}(q_{i1}(n) - q_{i2}(n))\textcolor{red}{|}                                                                                              \tag{B.7}
            \end{align}
            其中，$Q_{x|z}(m,n)$表示矩阵$Q_{x|z}$的第m行第n列的元素，$q_{i1}(n)$表示向量$q_{i1}$的第n个元素。
            
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_proof_ctr_zh")
    return


def md_reference_zh():
    global g_latex_del

    with gr.Accordion(label="Reference", elem_classes="first_md", elem_id="reference"):

        gr.Markdown(
            r"""
            <a id="dpm" href="https://arxiv.org/abs/1503.03585"> [1] Deep Unsupervised Learning Using Nonequilibrium Thermodynami </a>

            <a id="ddpm" href="https://arxiv.org/abs/1503.03585"> [2] Denoising Diffusion Probabilistic Models </a>

            <a id="linear_transform" href="https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/03%3A_Distributions/3.07%3A_Transformations_of_Random_Variables"> [3] Linear Transformations of Random Variable </a>

            <a id="sum_conv" href="https://stats.libretexts.org/Bookshelves/Probability_Theory/Probability_Mathematical_Statistics_and_Stochastic_Processes_(Siegrist)/03%3A_Distributions/3.07%3A_Transformations_of_Random_Variables"> [4] Sums and Convolution </a>

            <a id="fixed_point" href="https://en.wikipedia.org/wiki/Banach_fixed-point_theorem"> [5] Banach fixed-point theorem </a>

            <a id="ctr" href="https://en.wikipedia.org/wiki/Contraction_mapping"> [6] Contraction mapping </a>

            <a id="mc_limit" href="https://stats.libretexts.org/Bookshelves/Probability_Theory/Book%3A_Introductory_Probability_(Grinstead_and_Snell)/11%3A_Markov_Chains/11.04%3A_Fundamental_Limit_Theorem_for_Regular_Chains"> [7] Fundamental Limit Theorem for Regular Chains </a>

            <a id="mc_basic_p6" href="http://galton.uchicago.edu/~lalley/Courses/312/MarkovChains.pdf"> [8] Markov Chain:Basic Theory - Proposition 6 </a>
            
            <a id="fp_converse" href="https://arxiv.org/abs/1702.07339"> [9] A Converse to Banach's Fixed Point Theorem and its CLS Completeness </a>

            <a id="ce_kl" href="https://en.wikipedia.org/wiki/Cross-entropy#Cross-entropy_minimization"> [10] Cross-entropy minimization </a>
            
            <a id="deconv_1" href="https://thewolfsound.com/deconvolution-inverse-convolution/"> [11] Deconvolution Using Frequency-Domain Division </a>
            
            <a id="deconv_2" href="https://www.strollswithmydog.com/deconvolution-by-division-in-the-frequency-domain/"> [12] deconvolution-by-division-in-the-frequency-domain </a>
            
            <a id="mc_basic_t7" href="http://galton.uchicago.edu/~lalley/Courses/312/MarkovChains.pdf"> [13] Markov Chain:Basic Theory - Theorem 7 </a>
            
            <a id="mc_basic_d4" href="http://galton.uchicago.edu/~lalley/Courses/312/MarkovChains.pdf"> [14] Markov Chain:Basic Theory - Definition 4 </a>
            
            <a id="vdm" href="https://arxiv.org/pdf/2107.00630"> [15] Variational Diffusion Models </a>
            
            <a id="entropy" href="https://en.wikipedia.org/wiki/Entropy"> [16] Entropy </a>
            
            <a id="cond_entropy" href="https://en.wikipedia.org/wiki/Conditional_entropy"> [17] Conditional Entropy </a>
            
            <a id="dsm" href="https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport_1358_v1.pdf"> [18] A Connection Between Score Matching and Denoising autoencoders </a>
            
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_reference_zh")

    return


def md_about_zh():
    global g_latex_del

    with gr.Accordion(label="About", elem_classes="first_md", elem_id="about"):

        gr.Markdown(
            r"""
            <b>APP</b>: 本Web APP是使用Gradio开发，并部署在HuggingFace。由于资源有限(2核，16G内存)，所以可能会响应较慢。为了更好地体验，建议从<a href="https://github.com/blairstar/The_Art_of_DPM">github</a>复制源代码，在本地机器运行。本APP只依赖Gradio, SciPy, Matplotlib。
            
            <b>Author</b>: 郑镇鑫，资深视觉算法工程师，十年算法开发经历，曾就职于腾讯京东等互联网公司，目前专注于视频生成(类似Sora)。
            
            <b>Email</b>: blair.star@163.com 。
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_about_zh")

    return


def run_app():
    
    # with gr.Blocks(css=g_css, js="() => insert_special_formula() ", head=js_head) as demo:
    with gr.Blocks(css=g_css, js="() => {insert_special_formula(); write_markdown();}", head=js_head) as demo:
        md_introduction_zh()

        md_transform_zh()

        md_likelihood_zh()

        md_posterior_zh()

        md_forward_process_zh()

        md_backward_process_zh()
        
        md_fit_posterior_zh()
        
        md_posterior_transform_zh()
        
        md_deconvolution_zh()
        
        md_cond_kl_zh()
        
        md_proof_ctr_zh()

        md_reference_zh()
         
        md_about_zh()
        
    demo.launch(allowed_paths=["/"])
    
    return


if __name__ == "__main__":
    run_app()