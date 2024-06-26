
import gradio as gr
from Misc import g_css, js_head, g_latex_del
js_head += """ <script type="text/javascript" src="https://cdn.jsdelivr.net/npm/katex@0.15.3/dist/katex.min.js" integrity="sha384-0fdwu/T/EQMsQlrHCCHoH10pkPLlKA1jL5dFyUOvB3lfeT2540/2g6YgSi2BL14p" crossorigin="anonymous"></script> """



def md_introduction_en():
    global g_latex_del

    with gr.Accordion(label="0. Introduction", elem_classes="first_md", elem_id="introduction"):
        
        gr.Markdown(
            r"""
            The Diffusion Probability Model[\[1\]](#dpm)[\[2\]](#ddpm) is currently the main method used in image and video generation, but due to its abstruse theory, many engineers are unable to understand it well. This article will provide a very easy-to-understand method to help readers grasp the principles of the Diffusion Model. Specifically, it will illustrate the Diffusion Model using examples of one-dimensional random variables in an interactive way, explaining several interesting properties of the Diffusion Model in an intuitive manner.
            
            The diffusion model is a probabilistic model. Probabilistic models mainly offer two functions: calculating the probability of a given sample appearing; and generating new samples. The diffusion model focuses on the latter aspect, facilitating the production of new samples, thus realizing the task of **generation**.
            
            The diffusion model differs from general probability models (such as GMM), which directly models the probability distribution of random variables. The diffusion model adopts an indirect approach, which utilizes **random variable transform**(shown in Figure 1a) to gradually convert the data distribution (the probability distribution to be modeled) into the **standard normal distribution**, and meanwhile models the posterior probability distribution corresponding to each transformation (Figure 1b-c). Upon obtaining the final standard normal distribution and the posterior probability distributions, one can generate samples of each random variable $Z_T \ldots Z_2,Z_1,X$ in reverse order through <b>Ancestral Sampling</b>. Simultaneously, initial data distribution $q(x)$ can be determined by employing Bayes theorem and the total probability theorem.
            
            One might wonder: indirect methods require modeling and learning T posterior probability distributions, while direct methods only need to model one probability distribution, Why would we choose the indirect approach? Here's the reasoning: the initial data distribution might be quite complex and hard to represent directly with a probability model. In contrast, the complexity of each posterior probability distribution in indirect methods is significantly simpler, allowing it to be approximated by simple probability models. As we will see later, given certain conditions, posterior probability distributions can closely resemble Gaussian distributions, thus a simple conditional Gaussian model can be used for modeling.
            
            <center> <img id="en_fig1" src="file/fig1.png" width="820" style="margin-top:12px"/> </center>
            <center> Figure 1: Diffusion probability model schematic </center>
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_introduction_en")
    return


def md_transform_en():
    global g_latex_del

    with gr.Accordion(label="1. How To Transform", elem_classes="first_md", elem_id="transform"):
        
        gr.Markdown(
            r"""
            To transform the initial data distribution into a simple standard normal distribution, the diffusion model uses the following transformation method: 
            \begin{align}
                Z = \sqrt{\alpha} X +  \sqrt{1-\alpha}\epsilon \qquad where \quad \alpha < 1, \quad \epsilon \sim \mathcal{N}(0, I)  \tag{1.1}
            \end{align}
            where $X\sim q(x)$is any random variable，$Z\sim q(Z)$ is the transformed random variable。

            This transformation can be divided into two sub-transformations。
            
            The first sub-transformation performs a linear transformation ($\sqrt{\alpha}X$) on the random variable $X$. According to the conclusion of the literature[\[3\]](#linear_transform), the linear transformation makes the probability distribution of $X$ **narrower and taller**, and the extent of **narrowing and heightening** is directly proportional to the value of $\alpha$. 
            
            This can be specifically seen in <a href="#demo_1">Demo 1</a>, where the first figure depicts a randomly generated one-dimensional probability distribution, and the second figure represents the probability distribution after the linear transformation. It can be observed that the curve of the third figure has become **narrower and taller** compared to the first image. Readers can experiment with different $\alpha$ to gain a more intuitive understanding.
            
            The second sub-transformation is **adding independent random noise**($\sqrt{1-\alpha}\epsilon$). According to the conclusion of the literature[\[4\]](#sum_conv), **adding independent random variables** is equivalent to performing convolution on the two probability distributions. Since the probability distribution of random noise is Gaussian, it is equivalent to performing a **Gaussian Blur** operation. After blurring, the original probability distribution will become smoother and more similar to the standard normal distribution. The degree of blurring is directly proportional to the noise level ($\sqrt{1-\alpha}$).
            
            For specifics, one can see <a href="#demo_1">Demo 1</a>, where the first figure is a randomly generated one-dimensional probability distribution, and the third  figure is the result after the transformation. It can be seen that the transformed probability distribution curve is smoother and there are fewer corners. The readers can test different $\alpha$ values to feel how the noise level affect the shape of the probability distribution. The last figure is the result after applying all two sub-transformations.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_transform_en")
    return


def md_likelihood_en():
    global g_latex_del

    with gr.Accordion(label="2. Likelihood of The Transform", elem_classes="first_md", elem_id="likelihood"):

        gr.Markdown(
            r"""
            From the transformation method (equation 1.1), it can be seen that the probability distribution of the forward conditional probability $q(z|x)$ is a Gaussian distribution, which is only related to the value of $\alpha$, regardless of the probability distribution of $q(x)$. 
            \begin{align}
                q(z|x) &= \mathcal{N}(\sqrt{\alpha}x,\ 1-\alpha)    \tag{2.1}
            \end{align}
            It can be understood by concrete examples in <a href="#demo_2">Demo 2</a>. The third figure depict the shape of $q(z|x)$. From the figure, a uniform slanting line can be observed. This implies that the mean of $q(z|x)$ is linearly related to x, and the variance is fixed. The magnitude of $\alpha$ will determine the width and incline of the slanting line.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_likelihood_en")
    return


def md_posterior_en():
    global g_latex_del

    with gr.Accordion(label="3. Posterior of The Transform", elem_classes="first_md", elem_id="posterior"):

        gr.Markdown(
            r"""
            The posterior probability distribution does not have a closed form, but its shape can be inferred approximately through some technique.

            According to Bayes formula, we have
            \begin{align}
                q(x|z) = \frac{q(z|x)q(x)}{q(z)}    \tag{3.1}
            \end{align}

            When $z$ is fixed, $q(z)$ is a constant, so $q(x|z)$ is a probability density function with respect to $x$, and its shape depends only on $q(z|x)q(x)$. 
            \begin{align}
                q(x|z)  \propto q(z|x)q(x) 	\qquad where\ z\ is\ fixed  \tag{3.2}
            \end{align}
            
            In fact, $q(z)=\int q(z|x)q(x)dx$, which means that $q(z)$ is the sum over $x$ of the function $q(z|x)q(x)$. Therefore, dividing $q(z|x)q(x)$ by $q(z)$ is equivalent to normalizing $q(z|x)q(x)$. 
            \begin{align}
                q(x|z) = \operatorname{Normalize}\big(q(z|x)q(x)\big)  \tag{3.3}
            \end{align}
            
            From Equation 2.1, we can see that $q(z|x)$ is a Gaussian distribution, so we have
            \begin{align}
                q(x|z)  &\propto \frac{1}{\sqrt{2\pi(1-\alpha)}}\exp{\frac{-(z-\sqrt{\alpha}x)^2}{2(1-\alpha)}}\ q(x)& 	\qquad &\text{where z is fixed}     \notag   \newline
                        &= \frac{1}{\sqrt{\alpha}}\frac{1}{\sqrt{2\pi\frac{1-\alpha}{\alpha}}}\exp{\frac{-(\frac{z}{\sqrt{\alpha}}-x)^2}{2\frac{1-\alpha}{\alpha}}}\ q(x)& 	     \notag     \newline
                        &= \frac{1}{\sqrt{\alpha}} \underbrace{\frac{1}{\sqrt{2\pi}\sigma}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}}_{\text{GaussFun}}\ q(x)& \qquad &\text{where}\ \mu=\frac{z}{\sqrt{\alpha}}\quad \sigma=\sqrt{\frac{1-\alpha}{\alpha}}   \tag{3.4}
            \end{align}
            
            It can be observed that the <b>GaussFun</b> part is a Gaussian function of $x$, with a mean of $\frac{z}{\sqrt{\alpha}}$ and a variance of $\sqrt{\frac{1-\alpha}{\alpha}}$, so the shape of $q(x|z)$ is determined by **the product of GaussFun and q(x)**.
            
            According to the characteristics of <em>multiplication</em>, the characteristics of the shape of the $q(x|z)$ function can be summarized. 
            
            <ul>
            <li>The support set of $q(x|z)$ should be contained within the support set of GaussFun. The support set of GaussFun is a hypersphere, centered at the mean $\mu$ with a radius of approximately 3 times the standard deviation $\sigma$. </li>
            
            <li>When the variance of the Gaussian function is small (small noise), or when $q(x)$ changes linearly, the shape of $q(x|z)$ will approximate to the Gaussian function, and have a simpler function form, which is convenient for modeling and learning.</li>
             
            <li>When the variance of the Gaussian function is large (large noise), or when $q(x)$ changes drastically, the shape of $q(x|z)$ will be more complex, and greatly differ from a Gaussian function, which makes it difficult to model and learn.</li>
            </ul>
            
            <a href="#approx_gauss">Appendix B</a> provides a more rigorous analysis. When $\sigma$ satisfies certain conditions, $q(x|z)$ approximates to Gaussiani distribution.
            
            The specifics can be seen in <a href="#demo_2">Demo 2</a>. The fourth figure present the shape of the posterior $q(x|z)$, which shows an irregular shape and resembles a curved and uneven line. As $\alpha$ increases (noise decreases), the curve tends to be uniform and straight. Readers can adjust different $\alpha$ values and observe the relationship between the shape of posterior and the level of noise. In the last figure, the $\textcolor{blue}{\text{blue dash line}}$ represents $q(x)$, the $\textcolor{green}{\text{green dash line}}$ represents <b>GaussFun</b> in the equation 3.4, and the $\textcolor{orange}{\text{orange curve}}$ represents the result of multiplying the two function and normalizing it, which is the posterior probability $q(x|z=fixed)$ under a fixed z condition. Readers can adjust different values of z to observe how the fluctuation of $q(x)$ affect the shape of the posterior probability $q(x|z)$.
            
            The posterior $q(x|z)$ under two special states are worth considering.
            <ul>
            <li>As $\alpha \to 0$, the variance of <b>GaussFun</b> tends to <b>$\infty$</b>, and GaussFun almost becomes a uniform distribution over a very large support set, and the result of multiplying $q(x)$ by the uniform distribution is still $q(x)$, therefore, $q(x|z)$ for different $z$ almost become identical, and almost the same as $q(x)$. Readers can set $\alpha$ to 0.001 in <a href="#demo_2">Demo 2</a> to observe the specific results.</li>
                
            <li>As $\alpha \to 1$, the variance of <b>GaussFun</b> tends to <b>$0$</b>, The $q(x|z)$ for different $z$ values contract into a series of <em>Dirac delta functions</em> with different offsets equalling to $z$. However, there are some exceptions. When there are regions where $q(x)$ is zero, the corresponding $q(x|z)$ will no longer be a Dirac <em>delta function</em>, but a zero function. Readers can set $\alpha$ to 0.999 in <a href="#demo_2">Demo 2</a> to observe the specific results.</li>
            </ul>
            
            There is one point to note. when $\alpha \to 0$, the mean of GaussFun corresponding for larger $z$ values($\mu = \frac{z}{\sqrt{\alpha}}$) also increases sharply. This means that GaussFun is located farther from the support of $q(x)$. In this case, the "uniformity" of the part of GaussFun corresponding to the support of $q(x)$ will slightly decrease, thereby slightly reducing the similarity between $q(x|z)$ and $q(x)$. However, this effect will further diminish as $\alpha$ decreases. Readers can observe this effect in <a href="#demo_2">Demo 2</a>. Set $\alpha$ to 0.001, and you will see a slight difference between $q(x|z=-2)$ and $q(x)$, but no noticeable difference between $q(x|z=0)$ and $q(x)$.
            
            Regarding the "uniformity" of the Gaussian function, there are two characteristics: the larger the standard deviation, the greater the uniformity; the farther away from the mean, the smaller the uniformity.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_posterior_en")
    return


def md_forward_process_en():
    global g_latex_del

    title = "4. Transform Data Distribution To Normal Distribution"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="forward_process"):

        gr.Markdown(
            r"""
            For any arbitrary data distribution $q(x)$, the transform(equation 2.1) in section 2 can be continuously applied(equation 4.1~4.4). As the number of transforms increases, the output probability distribution will become increasingly closer to the standard normal distribution. For more complex data distributions, more iterations or larger noise are needed.
            
            Specific details can be observed in <a href="#demo_3_1">Demo 3.1</a>. The first figure illustrates a randomly generated one-dimensional probability distribution. After seven transforms, this distribution looks very similar to the standard normal distribution. The degree of similarity increases with the number of iterations and the level of the noise. Given the same degree of similarity, fewer transforms are needed if the noise added at each step is larger (smaller $\alpha$ value). Readers can try different $\alpha$ values and numbers of transforms to see how similar the final probability distribution is.
            
            The complexity of the initial probability distribution tends to be high, but as the number of transforms increases, the complexity of the probability distribution $q(z_t)$ will decrease. As concluded in section 4, a more complex probability distribution corresponds to a more complex posterior probability distribution. Therefore, in order to ensure that the posterior probability distribution is more similar to the Conditional Gaussian function (easier to learn), a larger value of $\alpha$ (smaller noise) should be used in the initial phase, and a smaller value of $\alpha$ (larger noise) can be appropriately used in the later phase to accelerate the transition to the standard normal distribution.
            
            In the example of <a href="#demo_3_1">Demo 3.1</a>, it can be seen that as the number of transforms increases, the corners of $q(z_t)$ become fewer and fewer. Meanwhile, the slanting lines in the plot of the posterior probability distribution $q(z_{t-1}|z_t)$ become increasingly straight and uniform, resembling more and more the conditional Gaussian distribution. 

            \begin{align}
                Z_1   &= \sqrt{\alpha_1} X + \sqrt{1-\alpha_1}\epsilon_1 			\tag{4.1}   \newline
                Z_2   &= \sqrt{\alpha_2} Z_1 + \sqrt{1-\alpha_2}\epsilon_2 			\tag{4.2}   \newline
                      &\dots														\notag      \newline
                Z_{t} &= \sqrt{\alpha_t}Z_{t-1} + \sqrt{1-\alpha_t}\epsilon_{t}	    \tag{4.3}   \newline
                      &\dots													    \notag      \newline
                Z_{T} &= \sqrt{\alpha_T}Z_{T-1} + \sqrt{1-\alpha_T}\epsilon_{T}	    \tag{4.4}   \newline
                      &where \quad \alpha_t < 1   \qquad t\in {1,2,\dots,T}         \notag
            \end{align}

            By substituting Equation 4.1 into Equation 4.2, and utilizing the properties of Gaussian distribution, we can derive the form of $q(z_2|x)$ 
            \begin{align}
                z_2 &= \sqrt{\alpha_2}(\sqrt{\alpha_1}x + \sqrt{1-\alpha_1}\epsilon_1) + \sqrt{1-\alpha_2}\epsilon_2	    \tag{4.5}   \newline
                    &= \sqrt{\alpha_2\alpha_1}x + \sqrt{\alpha_2-\alpha_2\alpha_1}\epsilon_1 + \sqrt{1-\alpha_2}\epsilon_2  \tag{4.6}   \newline
                    &= \mathcal{N}(\sqrt{\alpha_1\alpha_2}x,\ 1-\alpha_1\alpha_2)                                           \tag{4.7}
            \end{align}
            
            In the same way, it can be deduced recursively that
            \begin{align}
                q(z_t|x) &= \mathcal{N}(\sqrt{\alpha_1\alpha_2\cdots\alpha_t}x,\ 1-\alpha_1\alpha_2\cdots\alpha_t) = \mathcal{N}(\sqrt{\bar{\alpha_t}}x,\ 1-\bar{\alpha_t})  \qquad where\ \bar{\alpha_t} \triangleq \prod_{j=1}^t\alpha_j      \tag{4.8}
            \end{align}
            
            Comparing the forms of Equation 4.8 and Equation 2.1, it can be found that their forms are completely consistent. 
            
            If we only focus on the relationship between the initial and final random variables, then a sequence of t small transforms can be replaced by one large transform, and the $\alpha$ of the large transform is the accumulation of the $\alpha$ from each small transform, because the joint probability distributions corresponding to both types of transforms are the same.
            
            Readers can perform an experiment in <a href="#demo_3_1">Demo 3.1</a> using the same input distribution $q(x)$ but with two different transform methods: 1) using three transformations, each with $\alpha$ equal to 0.95; 2) using a single transform with $\alpha$ set to 0.857375. Perform the transformations separately and then compare the two resulting distributions. You will see that the two distributions are identical.
             
            In the DDPM[\[2\]](#ddpm) paper, the authors used 1000 steps (T=1000) to transform the data distribution $q(x)$ to $q(z_T)$. The probability distribution of $q(z_T|x)$ is as follows:
            \begin{align}
                q(z_T|x) &= \mathcal{N}(0.00635\ x,\ 0.99998)    \tag{4.9}
            \end{align}
            
            If only considering the joint distribution $q(x, z_T)$, a single transformation can also be used as a substitute, which is as follows:
            \begin{align}
                Z_T = \sqrt{0.0000403}\ X + \sqrt{1-0.0000403}\ \epsilon = 0.00635\ X + 0.99998\ \epsilon 			 \tag{4.10}
            \end{align}
            It can be seen that, after applying two transforms, the transformed distributions $q(z_T|x)$ are the same. Thus, $q(x,z_T)$ is also the same. 
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_forward_process_en")
    return


def md_backward_process_en():
    global g_latex_del

    title = "5. Restore Data Distribution From Normal Distribution"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="backward_process"):

        gr.Markdown(
            r"""
            If the final probability distribution $q(z_T)$ and the posterior probabilities of each transform $q(x|z),q(z_{t-1}|z_t)$ are known, the data distribution $q(x)$ can be recovered through the Bayes Theorem and the Law of Total Probability, as shown in equations 5.1~5.4. When the final probability distribution $q(z_T)$ is very similar to the standard normal distribution, the standard normal distribution can be used as a substitute.
            
            Specifics can be seen in <a href="#demo_3_2">Demo 3.2</a>. In the example, $q(z_T)$ substitutes $\mathcal{N}(0,1)$, and the error magnitude is given through JS Divergence. The restored probability distribution $q(z_t)$ and $q(x)$ are identified by the $\textcolor{green}{\text{green curve}}$, and the original probability distribution is identified by the $\textcolor{blue}{\text{blue curve}}$. It can be observed that the data distribution $q(x)$ can be well restored, and the error (JS Divergence) will be smaller than the error caused by the standard normal distribution replacing $q(z_T)$.
            \begin{align}
                q(z_{T-1}) &= \int q(z_{T-1},z_T)dz_T = \int q(z_{T-1}|z_T)q(z_T)dz_T               	    \tag{5.1}   \newline
                           & \dots	                                                                        \notag      \newline
                q(z_{t-1}) &= \int q(z_{t-1},z_t)dz_t = \int q(z_{t-1}|z_t)q(z_t)dz_t                       \tag{5.2}   \newline
                           & \dots	                                                                        \notag      \newline
                q(z_1)     &= \int q(z_1,z_2) dz_1    = \int q(z_1|z_2)q(z_2)dz_2                           \tag{5.3}   \newline
                q(x)       &= \int q(x,z_1) dz_1      = \int q(x|z_1)q(z_1)dz_1                             \tag{5.4}   \newline
            \end{align}
            In this article, the aforementioned transform is referred to as the <b>Posterior Transform</b>. For example, in equation 5.4, the input of the transform is the probability distribution function $q(z_1)$, and the output is the probability distribution function $q(x)$.The entire transform is determined by the posterior $q(x|z_1)$. This transform can also be considered as the linear weighted sum of a set of basis functions, where the basis functions are $q(x|z_1)$ under different $z_1$, and the weights of each basis function are $q(z_1)$. Some interesting properties of this transform will be introduced in <a href="#posterior_transform">Section 7</a>.
            
            In <a href="#posterior">Section 3</a>, we have considered two special posterior probability distributions. Next, we analyze their corresponding <em>posterior transforms</em>.
            <ul>
                <li> When $\alpha \to 0$, the $q(x|z)$ for different $z$ are almost the same as $q(x)$. In other words, the basis functions of linear weighted sum are almost the same. In this state, no matter how the input changes, the output of the transformation is always $q(x)$.</li>
                <li> When $\alpha \to 1$, the $q(x|z)$ for different $z$ values becomes a series of Dirac delta functions and zero functions. In this state, as long as the <em>support</em> of the input distribution is included in the <em>support set</em> of $q(x)$, the output of the transformation will remain the same with the input.</li>
            </ul>
            
            In <a href="#forward_process">Section 4</a>, it is mentioned that the 1000 transformations used in the DDPM[\[2\]](#ddpm) can be represented using a single transformation
            \begin{align}
                Z_T = \sqrt{0.0000403}\ X + \sqrt{1-0.0000403}\ \epsilon = 0.00635\ X + 0.99998\ \epsilon 			 \tag{5.5}
            \end{align}
            
            Since $\alpha=0.0000403$ is very small, the corresponding standard deviation of GaussFun (Equation 3.4) reaches 157.52. If we constrain the support of $q(x)$ within the unit hypersphere ($\lVert x \rVert_2 < 1$), then for $z_T$ in the range $[-2, +2]$, each corresponding $q(x|z_T)$ is very similar to $q(x)$. In this state, for the posterior transform of $q(x|z_T)$, regardless of the shape of the input distribution, as long as the support set is within the range $[-2,+2]$, the output distribution will be $q(x)$.
            
            <b>Furthermore, we can conclude that in the DPM model, if the support of $q(x)$ is finite and the signal-to-noise ratio of the final variable $Z_T$ is sufficiently high, the process of restoring $q(x)$ can use any distribution; it doesn't necessarily have to use the standard normal distribution.</b>
            
            Readers can conduct a similar experiment themselves. In <a href="#demo_3_1">Demo 3.1</a>, set <em>start_alpha</em> to 0.25, <em>end_alpha</em> to 0.25, and <em>step</em> to 7. At this point, $q(z_7)=\sqrt{0.000061}X + \sqrt{1-0.000061} \epsilon$, which is roughly equivalent to DDPM's $q(z_T)$. Click on <b>apply</b> to perform the forward transform (plotted using $\textcolor{blue}{\text{blue curves}}$), which prepares for the subsequent restoring process. In <a href="#demo_3_2">Demo 3.2</a>, set the <em>noise_ratio</em> to 1, introducing 100% noise into the <em>tail distribution</em> $q(z_7)$. Changing the value of <em>nose_random_seed</em> will change the distribution of noise. Deselect <em>backward_pdf</em> to reduce screen clutter. Click on <b>apply</b> to restore $q(x)$ through posterior transform. You will see that, no matter what the shape of input $q(z_7)$ may be, the restored $q(x)$ is always exactly the same as the original $q(x)$. The JS Divergence is zero. The restoration process is plotted using a $\textcolor{red}{\text{red curve}}$.
            
            There is another point worth noting. In deep learning tasks, it is common to scale each dimension of the input within the range [-1, 1], which means within a unit hypercube. The maximum Euclidean distance between any two points in a unit hypercube increases with the dimensionality. For example, in one dimension, the maximum distance is $2$, two dimensions is $2\sqrt{2}$, three dimensions is $2\sqrt{3}$, and n dimensions is $2\sqrt{n}$. Therefore, for data with higher dimensions, the variable $Z_T$ needs a higher signal-to-noise ratio to allow the starting distribution of the recovery process to accept any distribution.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_backward_process_en")
    return


def md_fit_posterior_en():
    global g_latex_del

    title = "6. Fitting Posterior With Conditional Gaussian Model"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="fit_posterior"):

        gr.Markdown(
            r"""
            From the front part of <a href="#posterior">Section 3</a>, it is known that the posterior probability distributions are unknown and related to $q(x)$. Therefore, in order to recover the data distribution or sample from it, it is necessary to learn and estimate each posterior probability distribution.
            
            From the latter part of <a href="#posterior">Section 3</a>, it can be understood that when certain conditions are met, each posterior probability distribution $q(x|z), q(z_{t-1}|z_t)$ approximates the Gaussian probability distribution. Therefore, by constructing a set of conditional Gaussian probability models $p(x|z), p(z_{t-1}|z_t)$, we can learn to fit the corresponding $q(x|z), q(z_{t-1}|z_t)$.
            
            Due to the limitations of the model's representative and learning capabilities, there will be certain errors in the fitting process, which will further impact the accuracy of restored $q(x)$. The size of the fitting error is related to the complexity of the posterior probability distribution. As can be seen from <a href="#posterior">Section 3</a>, when $q(x)$ is more complex or the added noise is large, the posterior probability distribution will be more complex, and it will differ greatly from the Gaussian distribution, thus leading to fitting errors and further affecting the restoration of $q(x)$.
            
            Refer to <a href="#demo_3_3">Demo 3.3</a> for the specifics. The reader can test different $q(x)$ and $\alpha$, observe the fitting degree of the posterior probability distribution $q(z_{t-1}|z_t)$ and the accuracy of restored $q(x)$. The restored probability distribution is ploted with $\textcolor{orange}{\text{orange}}$, and the error is also measured by JS divergence.
             
            Regarding the objective function for fitting, similar to other probability models, the cross-entropy loss can be optimized to make $p(z_{t-1}|z_t)$ approaching $q(z_{t-1}|z_t)$. Since $(z_{t-1}|z_t)$ is a conditional probability, it is necessary to fully consider all conditions. This can be achieved by averaging the cross-entropy corresponding to each condition weighted by the probability of each condition happening. The final form of the loss function is as follows.
            \begin{align}
                loss &= -\int q(z_t) \overbrace{\int q(z_{t-1}|z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}}^{\text{Cross Entropy}}\ dz_t     \tag{6.1}     \newline
                     &= -\iint q(z_{t-1},z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                     \tag{6.2} 
            \end{align}
            
            KL divergence can also be optimized as the objective function. KL divergence and cross-entropy are equivalent[\\[10\\]](#ce_kl)
            <span id="en_fit_0">
                loss &= \int q(z_t) KL(q(z_{t-1}|z_t) \Vert \textcolor{blue}{p(z_{t-1}|z_t)})dz_t                                                             \tag{6.3}      \newline
                     &= \int q(z_t) \int q(z_{t-1}|z_t) \log \frac{q(z_{t-1}|z_t)}{\textcolor{blue}{p(z_{t-1}|z_t)}} dz_{t-1} dz_t                            \tag{6.4}      \newline 
                     &= -\int q(z_t)\ \underbrace{\int q(z_{t-1}|z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}}{underline}{\text{Cross Entropy}}\ dz_t + \underbrace{\int q(z_t) \int q(z_{t-1}|z_t) \log q(z_{t-1}|z_t)}{underline}{\text{Is Constant}} dz  \tag{6.5}
            </span>

            The integral in equation 6.2 does not have a closed form and cannot be directly optimized. The Monte Carlo integration can be used for approximate calculation. The new objective function is as follows:
            \begin{align}
                loss &= -\iint q(z_{t-1},z_t) \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                     \tag{6.6}      \newline
                     &\approx -\sum_{i=0}^N \log \textcolor{blue}{p(Z_{t-1}^i|Z_t^i)} \qquad where \quad (Z_{t-1}^i,Z_t^i) \sim q(z_{t-1},z_t)      \tag{6.7} 
            \end{align}

            The aforementioned samples $(Z_{t-1}^i,Z_t^i)$ follow a joint probability distribution $q(z_{t-1},z_t)$, which can be sampled via an <b>Ancestral Sampling</b>. The specific method is as follows: sample $X,Z_1,Z_2 \dots Z_{t-1},Z_t$ step by step through forward transforms (Formulas 4.1~4.4), and then reserve $(Z_{t-1},Z_t)$ as a sample. This sampling process is relatively slow. To speed up the sampling, we can take advantage of the known features of the probability distribution $q(z_t|x)$ (Formula 4.8). First, sample $X$ from $q(x)$, then sample $Z_{t-1}$ from $q(z_{t-1}|x)$, and finally sample $Z_t$ from $q(z_t|z_{t-1})$. Thus, a sample $(Z_{t-1},Z_t)$ is obtained.
            
            Some people may question that the objective function in Equation 6.3 seems different from those in the DPM[\\[1\\]](#dpm) and DDPM[\\[2\\]](#ddpm) papers. In fact, these two objective functions are equivalent, and the proof is given below. 

            For <b>Consistent Terms</b>, the proof is as follows:

            \begin{align}
                loss &= -\iint q(z_{t-1},z_t)\ \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                                                            \tag{6.8}           \newline
                     &= -\iint \int q(x)q(z_{t-1}, z_t|x)dx\ \log \textcolor{blue}{p(z_{t-1}|z_t)}dz_{t-1}dz_t                                                                              \tag{6.9}           \newline
                     &= \overbrace{\iint \int q(x)q(z_{t-1}, z_t|x) \log q(z_{t-1}|z_t,x)dxdz_{t-1}dz_t}^{\text{This Term Is Constant And Is Denoted As}\ \textcolor{orange}{C_1}}          \tag{6.10}          \newline
                     &\quad - \iint \int q(x)q(z_{t-1}, z_t|x) \log \textcolor{blue}{p(z_{t-1}|z_t)}dxdz_{t-1}dz_t - \textcolor{orange}{C_1}                                                \tag{6.11}          \newline
                     &= \iint \int q(x)q(z_{t-1},z_t|x) \log \frac{q(z_{t-1}|z_t,x)}{\textcolor{blue}{p(z_{t-1}|z_t)}}dxdz_{t-1}dz_t - \textcolor{orange}{C_1}                              \tag{6.12}          \newline
                     &= \iint q(x)q(z_t|x)\int q(z_{t-1}|z_t,x) \log \frac{q(z_{t-1}|z_t,x)}{\textcolor{blue}{p(z_{t-1}|z_t)}}dz_{t-1}\ dz_t dx - \textcolor{orange}{C_1}                  \tag{6.13}          \newline
                     &= \iint \ q(x)q(z_t|x) KL[q(z_{t-1}|z_t,x) \Vert \textcolor{blue}{p(z_{t-1}|z_t)}] dz_t dx - \textcolor{orange}{C_1}                                                         \tag{6.14}          \newline
                     &\propto \iint \ q(x)q(z_t|x) KL(q(z_{t-1}|z_t,x) \Vert \textcolor{blue}{p(z_{t-1}|z_t)}) dz_t dx                                                                             \tag{6.15}          \newline
            \end{align}

            In the above formula, the term $C_1$ is a fixed value, which does not contain parameters to be optimized. Here, $q(x)$ is a fixed probability distribution, and $q(z_{t-1}|z_t)$ is also a fixed probability distribution, whose specific form is determined by $q(x)$ and the coefficient $\alpha$.
            
            For the <b>Reconstruction Term</b>, it can be proven in a similar way.

            \begin{align}
                loss &= -\int q(z_1)\overbrace{\int q(x|z_1)\log \textcolor{blue}{p(x|z_1)}dx}^{\text{Cross Entropy}}\ dz_1     \tag{6.16}   \newline
                &= -\iint q(z_1,x)\log \textcolor{blue}{p(x|z_1)}dxdz_1                 \tag{6.17}   \newline
                &= -\int q(x)\int q(z_1|x)\log \textcolor{blue}{p(x|z_1)}dz_1\ dx       \tag{6.18}
            \end{align}
            
            Therefore, the objective function in equation 6.1 is equivalent with the DPM original objective function.
            
            Based on the conclusion of the Consistent Terms proof and the relationship between cross entropy and KL divergence, an interesting conclusion can be drawn:
            <span id="en_fit_1">
                \mathop{\min}{underline}{\textcolor{blue}{p}} \int q(z_t) KL(q(z_{t-1}|z_t) \Vert \textcolor{blue}{p(z_{t-1}|z_t)})dz_t  \iff  \mathop{\min}{underline}{\textcolor{blue}{p}} \iint \ q(z_t)q(x|z_t) KL(q(z_{t-1}|z_t,x) \Vert \textcolor{blue}{p(z_{t-1}|z_t)})dxdz_t       \tag{6.19}
            </span>
            By comparing the expressions on the left and right, it can be observed that the objective function on the right side includes an additional variable $X$ compared to the left side. At the same time, there is an additional integral with respect to $X$, with the occurrence probability of $X$, denoted as $q(x|z_t)$, serving as the weighting coefficient for the integral.
            
            Following a similar proof method, a more general relationship can be derived:
            <span id="en_fit_2">
                \mathop{\min}{underline}{\textcolor{blue}{p}}  KL(q(z) \Vert \textcolor{blue}{p(z)})  \iff  \mathop{\min}_{\textcolor{blue}{p}} \int \ q(x) KL(q(z|x) \Vert \textcolor{blue}{p(z)})dx         \tag{6.20}
            </span>
            A detailed derivation of this conclusion can be found in <a href="#cond_kl">Appendix A</a>.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_fit_posterior_en")
    return


def md_posterior_transform_en():
    global g_latex_del

    with gr.Accordion(label="7. Posterior Transform", elem_classes="first_md", elem_id="posterior_transform"):

        gr.Markdown(
            r"""
            </br>
            <h3 style="font-size:18px"> Non-expanding Mapping and Stationary Distribution </h3>
            \begin{align}
                q(x) &= \int q(x,z) dz = \int q(x|z)q(z)dz      \tag{7.1}
            \end{align}

            According to Corollary 1 and Corollary 2 in <a href="#non_expanping">Appendix B</a>, the posterior transform is a <b>non-expanding mapping</b>. This means that for any two probability distributions $q_{i1}(z)$ and $q_{i2}(z)$, after the posterior transform, the resulting distributions $q_{o1}(x)$ and $q_{o2}(x)$ will have a distance that is <b>always less than or equal to</b> the distance between $q_{i1}(x)$ and $q_{i2}(x)$. The distance here can be measured using KL Divergence or Total Variance.
            \begin{align}
                d(q_{o1}(z),\ q_{o2}(z)) \le d(q_{i1}(x),\ q_{i2}(x))                   \tag{7.2}
            \end{align}
            According to the analysis in <a href="#non_expanping">Appendix B</a>, the aforementioned equality does not hold in most cases and posterior transform becomes a <b>shrinkig mapping</b>. Furthermore, <b>the smaller $\alpha$ is (the more noise), the smaller $d(q_{o1},q_{o2})$ will be compared to $d(q_{i1},q_{i2})$</b>.
    
            Readers can refer to <a href="#demo_4_1">Demo 4.1</a>, where the first three figure present a transform process. The first figure is an arbitrary data distribution $q(x)$, the third figure is the transformed probability distribution, and second figure is the posterior probability distribution $q(x|z)$. You can change the random seed to generate a new data distribution$q(x)$, and adjust the value of $\alpha$ to introduce different degrees of noise.

            The last two figures show contraction of the transform. The fourth figure displays two randomly generated input distributions and their distance, $div_{in}$. The fifth figure displays the two output distributions after transform, with the distance denoted as $div_{out}$.
            
            Readers can change the input random seed to toggle different inputs. It can be observed from the figures that $div_{in}$ is always smaller than $div_{out}$ for any input. Additionally, if you change the value of $\alpha$, you will see that the smaller the $\alpha$(larger noise), the smaller the ratio of $div_{out}/div_{in}$，indicating a larger rate of contraction.
            
            According to the analysis in <a href="#stationary">Appendix C</a>: the posterior transform can be seen as a one-step jump of a Markov chain, and <b>when $q(x)$ and $\alpha$ meet certain conditions, this Markov chain will converge to a unique stationary distribution</b>. Additionally, numerous experiments have shown that <b>the stationary distribution is very similar to the data distribution $q(x)$, and the smaller $\alpha$ is, the more similar the stationary distribution is to $q(x)$</b>. Specifically, according to the conclusion in <a href="#backward_process">Section 5</a>, <b>when $\alpha \to 0$, after one step of transform, the output distribution will be $q(x)$, so the stationary distribution must be $q(x)$</b>.
            
            Readers can refer to <a href="#demo_4_2">Demo 4.2</a>, which illustrates an example of applying posterior transform iteratively. Choose an appropriate number of iterations, and click on the button of <em>Apply</em>, and the iteration process will be draw step by step. Each subplot shows the transformed output distribution($\textcolor{green}{\text{green curve}}$) from each transform, with the reference distribution $q(x)$ expressed as a $\textcolor{blue}{\text{blue curve}}$, as well as the distance $div$ between the output distribution and $q(x)$. It can be seen that as the number of iterations increases, the output distribution becomes more and more similar to $q(x)$, and will eventually stabilize near $q(x)$. For more complicated distributions, more iterations or greater noise may be required. The maximum number of iterations can be set to tens of thousands, but it'll take longer.
            
            For the one-dimensional discrete case, $q(x|z)$ is discretized into a matrix (denoted as $Q_{x|z}$), $q(z)$ is discretized into a vector (denoted as $\boldsymbol{q_i}$). The integration operation $\int q(x|z)q(z)dz$ is discretized into a **matrix-vector** multiplication operation, thus the posterior transform can be written as
            \begin{align}
                \boldsymbol{q_o} &= Q_{x|z}\ \boldsymbol{q_i} &             \quad\quad        &\text{1 iteration}         \tag{7.3}       \newline
                \boldsymbol{q_o} &= Q_{x|z}\ Q_{x|z}\ \boldsymbol{q_i} &    \quad\quad        &\text{2 iteration}         \tag{7.4}       \newline
                    & \dots &                                                                                             \notag          \newline
                \boldsymbol{q_o} &= (Q_{x|z})^n\ \boldsymbol{q_i} &         \quad\quad        &\text{n iteration}         \tag{7.5}       \newline
            \end{align}
            In order to better understand the property of the transform, the matrix $(Q_{x|z})^n$ is also plotted in <a href="#demo_4_2">Demo 4.2</a>. From the demo we can see that, as the iterations converge, the row vectors of the matrix $(Q_{x|z})^n$ will become a constant vector, that is, all components of the vector will be the same, which will appear as a horizontal line in the denisty plot.
            
            For a one-dimensional discrete Markov chain, the convergence rate is inversely related to the absolute value of the second largest eigenvalue of the transition probability matrix ($\lvert \lambda_2 \rvert$). The smaller $\lvert \lambda_2 \rvert$ is, the faster the convergence. Numerous experiments have shown that $\alpha$ has a clear linear relationship with $\lvert \lambda_2 \rvert$; the smaller $\alpha$ is, the smaller $\lvert \lambda_2 \rvert$ is. Therefore, <b>the smaller $\alpha$ (the greater the noise), the faster the convergence rate</b>. Specifically, when $\alpha \to 0$, according to the conclusion in <a href="#posterior">Section 3</a>, the posterior probability distributions corresponding to different $z$ tend to be consistent. Additionally, according to Theorem 21 in <a href="#non_neg_lambda">[21]</a>, $\lvert \lambda_2 \rvert$ is smaller than the L1 distance between any two posterior probability distributions corresponding to different $z$, so it can be concluded that $\lvert \lambda_2 \rvert \to 0$.
            
            
            </br>
            <h3 style="font-size:18px"> Anti-noise Capacity In Restoring Data Distribution</h3>
            
            From the above analysis, it can be seen that, in most cases, the <b>posterior transform</b> is a shrinking mapping, which means the following relationship
            
            \begin{align}
                d(q(x),\ q_o(x)) < d(q(z),\ q_i(z))         \tag{7.12}
            \end{align}
            
            Among them, $q(z)$ is the ideal input distribution, $q(x)$ is the ideal output distribution, $q(x) = \int q(x|z) q(z) dz$, $q_i(z)$ is any input distribution, and $q_o(x)$ is the transformed output distribution, $q_o(x) = \int q(x|z) q_i(z) dz$.
            
            The above equation indicates that the distance between the output distribution $q_o(x)$ and the ideal output distribution q(x) will always be <b>less than</b> the distance between the input distribution $q_i(z)$ and the ideal input distribution q(x). Hence, <b>the posterior transform naturally possesses a certain ability to resist noise </b>. This means that during the process of restoring $q(x)$(<a href="#backward_process">Section 5</a>), even if the <em>tail distribution</em> $q(z_T)$ contains some error, the error of the outputed distribution $q(x)$ will be smaller than the error of input after undergoing a series of transform.
            
            Refer specifically to <a href="#demo_3_2">Demo 3.2</a>, where by increasing the value of the <b>noise ratio</b>, noise can be added to the <em>tail distribution</em> $q(z_T)$. Clicking the "apply" button will gradually draw out the restoring process, with the restored distribution represented by a $\textcolor{red}{\text{red curve}}$, and the error size will be computed by the JS divergence. You will see that the error of restored $q(x)$ is always less than the error of $q(z_T)$.
            
            From the above discussion, it can be seen that the smaller $\alpha$ is (the larger the noise used in the transform), the greater the shrinking rate of the shrink mapping, and correspondingly, the stronger the error resistance capability. Specifically, when $\alpha \to 0$, the noise resistance capability becomes infinite, meaning that regardless of the magnitude of the error in the input, the output will always be $q(x)$.
            
            </br>
            <h3 style="font-size:18px"> Markov Chain Monte Carlo Sampling</h3>
            
            In DPM models, sampling is typically performed using <b>Ancestral Sampling</b>. From the analysis above, it can be inferred that when $\alpha$ is sufficiently small, the posterior transform will converge to $q(x)$. Therefore, sampling can be conducted using <b>Markov Chain Monte Carlo</b> (MCMC) methods, as depicted in Figure 7.1. In the figure, $\alpha$ represents a posterior transform with relatively large noise, where larger noise makes the steady-state distribution closer to the data distribution $q(x)$. However, as discussed in Section 3, posterior transform with larger noise are less favorable for fitting. Therefore, transform with larger noise are split into multiple transform with smaller noise.
            
            <center> <img src="file/7.1.png" width="1024" style="margin-top:12px"/> </center>
            <center> Figure 7.1: Markov Chain Monte Carlo Sampling</center>

            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_posterior_transform_en")
    return


def md_deconvolution_en():
    global g_latex_del

    title = "8. Can the data distribution be restored by deconvolution?"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="deconvolution"):
        gr.Markdown(
            r"""
            As mentioned in the <a href="#introduction">Section 1</a>, the transform of Equation 2.1 can be divided into two sub-transforms, the first one being a linear transform and the second being adding independent Gaussian noise. The linear transform is equivalent to a scaling transform of the probability distribution, so it has an inverse transformation. Adding independent Gaussian noise is equivalent to the execution of a convolution operation on the probability distribution, which can be restored through <b>deconvolution</b>. Therefore, theoretically, the data distribution $q(x)$ can be recovered from the final probability distribution $q(z_T)$ through <b>inverse linear transform</b> and <b>deconvolution</b>.
            
            However, in actuality, some problems do exist. Due to the extreme sensitivity of deconvolution to errors, having high input sensitivity, even a small amount of input noise can lead to significant changes in output[\[11\]](#deconv_1)[\[12\]](#deconv_2). Meanwhile, in the diffusion model, the standard normal distribution is used as an approximation to replace $q(z_T)$, thus, noise is introduced at the initial stage of restoring. Although the noise is relatively small, because of the sensitivity of deconvolution, the noise will gradually amplify, affecting the restoring.
            
            In addition, the infeasibility of <b>deconvolution restoring</b> can be understood from another perspective. Since the process of forward transform (equations 4.1 to 4.4) is fixed, the convolution kernel is fixed. Therefore, the corresponding deconvolution transform is also fixed. Since the initial data distribution $q(x)$ is arbitrary, any probability distribution can be transformed into an approximation of $\mathcal{N}(0,I)$ through a series of fixed linear transforms and convolutions. If <b>deconvolution restoring</b> is feasible, it means that a fixed deconvolution can be used to restore any data distribution $q(x)$ from the $\mathcal{N}(0,I)$ , this is clearly <b>paradoxical</b>. The same input, the same transform, cannot have multiple different outputs.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_deconvolution_en")
    return


def md_cond_kl_en():
    global g_latex_del

    title = "Appendix A Conditional KL Divergence"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="cond_kl"):
        gr.Markdown(
            r"""
            This section mainly introduces the relationship between <b>KL divergence</b> and <b>conditional KL divergence</b>. Before the formal introduction, we will briefly introduce the definitions of <b>Entropy</b> and <b>Conditional Entropy</b>, as well as the inequality relationship between them, in preparation for the subsequent proof. 
            
            </br>
            <h3 style="font-size:20px">Entropy and Conditional Entropy</h3>
            For any two random variables $Z, X$, the <b>Entropy</b> is defined as follows<a href="#entropy">[16]</a>：
            \begin{align}
               \mathbf{H}(Z) = \int -p(z)\log{p(z)}dz    \tag{A.1}
            \end{align}
            The <b>Conditional Entropy</b> is defined as followed <a href="#cond_entropy">[17]</a>：
            \begin{align}
               \mathbf{H}(Z|X) = \int p(x) \overbrace{\int -p(z|x)\log{p(z|x)}dz}^{\text{Entropy}}\ dx    \tag{A.2}
            \end{align}
            The following inequality relationship exists between the two：
            \begin{align}
               \mathbf{H}(Z|X) \le \mathbf{H}(Z)         \tag{A.3}
            \end{align}
            It is to say that <b>the Conditional Entropy is always less than or equal to the Entropy</b>, and they are equal only when X and Z are independent. The proof of this relationship can be found in the literature <a href="#cond_entropy">[17]</a>.
            
            </br>
            <h3 style="font-size:20px">KL Divergence and Conditional KL Divergence</h3>
            In the same manner as the definition of Conditional Entropy, we introduce a new definition, <b>Conditional KL Divergence</b>, denoted as $KL_{\mathcal{C}}$. Since KL Divergence is non-symmetric, there exist two forms as follows. 
            \begin{align}
               KL_{\mathcal{C}}(q(z|x) \Vert \textcolor{blue}{p(z)}) = \int \ q(x) KL(q(z|x) \Vert \textcolor{blue}{p(z)})dx                        \tag{A.4}     \newline
               KL_{\mathcal{C}}(q(z) \Vert \textcolor{blue}{p(z|x)}) = \int \ \textcolor{blue}{p(x)} KL(q(z) \Vert \textcolor{blue}{p(z|x)})dx      \tag{A.5}
            \end{align}

            Similar to Conditional Entropy, there also exists a similar inequality relationship for <b>both forms of Conditional KL Divergence</b>:
            \begin{align}
               KL_{\mathcal{C}}(q(z|x) \Vert \textcolor{blue}{p(z)}) \ge KL(q(z) \Vert \textcolor{blue}{p(z)})        \tag{A.6}   \newline
               KL_{\mathcal{C}}(q(z) \Vert \textcolor{blue}{p(z|x)}) \ge KL(q(z) \Vert \textcolor{blue}{p(z)})        \tag{A.7}
            \end{align}
            It is to say that <b>the Conditional KL Divergence is always less than or equal to the KL Divergence</b>, and they are equal only when X and Z are independent.

            The following provides proofs for the conclusions on Equation A.5 and Equation A.6 respectively.

            For equation A.6, the proof is as follows:
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
            In this context, equation A.15 applies the conclusion that <b>Conditional Entropy is always less than or equal to Entropy</b>. Thus, the relationship in equation A.6 is derived. 

            For equation A.6, the proof is as follows:
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
            Thus, the relationship in equation A.7 is obtained.

            Another <b>important conclusion</b> can be drawn from equation A.15.

            The KL Divergence is often used to fit the distribution of data. In this scenario, the distribution of the data is denoted by $q(z)$ and the parameterized model distribution is denoted by $\textcolor{blue}{p_\theta(z)}$. During the optimization process, since both $q(z|x)$ and $q(x)$ remain constant, the term $\mathbf{H}(Z) - \mathbf{H}(Z|X)$ in Equation A.15 is a constant. Thus, the following relationship is obtained:
            <span id="en_cond_kl_2">
                \mathop{\min}{underline}{\textcolor{blue}{p_\theta}}  KL(q(z) \Vert \textcolor{blue}{p_\theta(z)})  \iff  \mathop{\min}{underline}{\textcolor{blue}{p_\theta}} \int \ q(x) KL(q(z|x) \Vert \textcolor{blue}{p_\theta(z)})dx   \tag{A.25}
            </span>

            Comparing the above relationship with <b>Denoised Score Matching</b> <a href="#dsm">[18]</a>(equation A.26), some similarities can be observed. Both introduce a new variable $X$, and substitute the targeted fitting distribution q(z) with q(z|x). After the substitution, since q(z|x) is a conditional probability distribution, both consider all conditions and perform a weighted sum using the probability of the conditions occurring, $q(x)$, as the weight coefficient.
            <span id="en_cond_kl_3">
                \mathop{\min}{underline}{\textcolor{blue}{\psi_\theta}} \frac{1}{2} \int q(z) \left\lVert \textcolor{blue}{\psi_\theta(z)} - \frac{\partial q(z)}{\partial z} \right\rVert^2 dz \iff  \mathop{\min}{underline}{\textcolor{blue}{\psi_\theta}} \int q(x)\ \overbrace{\frac{1}{2} \int q(z|x) \left\lVert \textcolor{blue}{\psi_\theta(z)} - \frac{\partial q(z|x)}{\partial z} \right\rVert^2 dz}^{\text{Score Matching of }q(z|x)}\ dx      \tag{A.26}
            </span>

            The operation of the above weighted sum is somewhat similar to <em> Elimination by Total Probability Formula </b>.
            \begin{align}
                q(z) = \int q(z,x) dx = \int q(x) q(z|x) dx     \tag{A.27}
            \end{align}
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_cond_kl_en")
    return


def md_approx_gauss_en():
    global g_latex_del

    title = "Appendix B When does the Posterior Approximate to Gaussian ?"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="approx_gauss"):
        gr.Markdown(
            r"""
            From equation 3.4, it can be seen that $q(x|z)$ takes the following form:
            \begin{align}
                q(x|z)  &=  \operatorname{Normalize} \Big(\  \frac{1}{\sqrt{2\pi}\sigma}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}\ q(x)\ \Big)& \qquad &\text{where}\ \mu=\frac{z}{\sqrt{\alpha}}\quad \sigma=\sqrt{\frac{1-\alpha}{\alpha}}       \tag{B.1}   \newline
                &\propto \underbrace{\frac{1}{\sqrt{2\pi}\sigma}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}}_{\text{GaussFun}}\ q(x)     \tag{B.2}
            \end{align}

            Below we will prove that if the following two assumptions are satisfied, $q(x|z)$ approximates a Gaussian distribution.
            <ul>
            <li>
            Assume that within the support of GaussFun, $q(x)$ undergoes linear changes. Expand $q(x)$ around the mean of GaussFun using a Taylor series. According to the properties of Taylor expansion, these assumptions can be satisfied when the standard deviation $\sigma$ of GaussFun is sufficiently small.
            \begin{align}
                q(x) &\approx  q(\mu) + \nabla_xq(\mu)(x-\mu)& \quad &\text{where}\quad q(\mu)\triangleq q(x)\bigg|_{x=\mu} \quad \nabla_xq(\mu)\triangleq \nabla_xq(x)\bigg|_{x=\mu}       \tag{B.3}   \newline
                      &= q(\mu)\big(1+ \frac{\nabla_xq(\mu)}{q(\mu)}(x-\mu)\big)&	    \tag{B.4}   \newline
                      &= q(\mu)\big(1+ \nabla_x\log{q(\mu)}(x-\mu)\big)& \quad &\text{where}\quad \nabla_x\log{q(\mu)}\triangleq \nabla_x\log{q(x)}\bigg|_{x=\mu}       \tag{B.5}
            \end{align}
            </li>
            <li>
            Assuming within the support of GaussFun, $\log\big(1+\nabla_x\log{q(\mu)}(x-\mu)\big)$ can be approximated by $\nabla_x\log{q(\mu)}(x-\mu)$. By expanding $\log(1+y)$ using Taylor series, according to the properties of Taylor expansion, when $\lVert y\rVert_2$ is small, $\log(1+y)$ can be approximated by $y$. When $\sigma$ is sufficiently small, $\lVert x-u\rVert_2$ will be small, and $\nabla_x\log{q(\mu)}(x-\mu)$will also be small, hence the above assumption can be satisfied. Generally, when $\nabla_x\log{q(\mu)}(x-\mu)<0.1$, the approximation error is small enough to be negligible.
            \begin{align}
                \log(1+y) &\approx \log(1+y)\bigg|_{y=0} + \nabla_y\log(1+y)\bigg|_{y=0}(y-0)       \tag{B.6}   \newline
                          &= y              \tag{B.7}
            \end{align}
            </li>
            </ul>
            Using the above two assumptions, $q(x|z)$ can be transformed into the following form：

            \begin{align}
                q(x|z) &\propto \frac{1}{\sqrt{2\pi}\sigma}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}\ q(x)        \tag{B.8}   \newline
                       &\approx \frac{1}{\sqrt{2\pi}\sigma}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}\  q(\mu)\big(1+ \nabla_x\log{q(\mu)}(x-\mu)\big)          \tag{B.9}   \newline
                       &= \frac{q(\mu)}{\sqrt{2\pi}\sigma}\exp\left(\frac{-(x-\mu)^2}{2\sigma^2}+\log\big(1+ \nabla_x\log{q(\mu)}(x-\mu)\big)\right)    \tag{B.10}  \newline
                       &\approx \frac{q(\mu)}{\sqrt{2\pi}\sigma}\exp\left(\frac{-(x-\mu)^2}{2\sigma^2}+\nabla_x\log{q(\mu)}(x-\mu)\right)               \tag{B.11}  \newline
                       &= \frac{q(\mu)}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2-2\sigma^2\nabla_x\log{q(\mu)}(x-\mu)}{2\sigma^2}\right)            \tag{B.12}  \newline
                       &= \frac{q(\mu)}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\big(x-\mu-\sigma^2\nabla_x\log{q(\mu)}\big)^2}{2\sigma^2}+\frac{\big(\sigma^2\nabla_x\log{q(\mu)}\big)^2}{2\sigma^2}\right)             \tag{B.13}   \newline
                       &= \exp\left(-\frac{\big(x-\mu-\sigma^2\nabla_x\log{q(\mu)}\big)^2}{2\sigma^2}\right) \underbrace{\frac{q(\mu)}{\sqrt{2\pi}\sigma} \exp\left( \frac{1}{2}\big(\sigma\nabla_x\log{q(\mu)}\big)^2\right)}_{\text{const}}       \tag{B.14}
            \end{align}

            Among them, Equation B.9 applies the conclusion of Assumption 1, and Equation B.11 applies the conclusion of Assumption 2. 

            The <em>const term</em> in Equation B.14 is constant and does not affect the shape of the function. Additionally, as can be seen from the above, $q(x|z)$ is self-normalizing. Therefore, $q(x|z)$ is a Gaussian probability density function with a mean of $\mu + \sigma^2 \nabla_x \log{q(\mu)}$ and a variance of $\sigma^2$.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_approx_gauss_en")

    return


def md_non_expanding_en():
    global g_latex_del

    title = "Appendix C Posterior Transform is a Non-expanding Mapping"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="non_expanding"):
        gr.Markdown(
            r"""
            <b>Corollary 1</b>

            Using KL Divergence as a metric, the transition transform of Markov chain is non-expanding<a href="#elem">[23]</a>, which means
            \begin{align}
                KL\big(p(x), q(x)\big) &\le KL\big(p(z), q(z)\big)          \tag{C.1} \newline
            \end{align}
            Here, $p(z)$ and $q(z)$ are arbitrary probability density functions, and $r(x|z)$ is the transition probability density function of the Markov chain. We have $p(x) = \int r(x|z)p(z)dz$ and $q(x) = \int r(x|z)q(z)dz$. 

            Proof：

            For the KL divergence of $p(x,z)$ and $q(x,z)$, the following relationship exists:
            \begin{align}
                KL\big(p(x,z), q(x,z)\big) &= \iint p(x,z)\log \frac{p(x,z)}{q(x,z)}dxdz    \tag{C.2} \newline
                & = \iint p(x,z)\log \frac{p(x)p(x|z)}{q(x)q(x|z)}dxdz                      \tag{C.3} \newline
                &= \iint p(x,z)\log \frac{p(x)}{q(x)}dxdz + \iint p(x,z) \log\frac{p(x|z)}{q(x|z)} dxdz     \tag{C.4} \newline
                &= \int \int p(x,z) dz\ \log \frac{p(x)}{q(x)}dx + \int p(z)\int p(x|z) \log\frac{p(x|z)}{q(x|z)} dx\ dz    \tag{C.5} \newline
                &= KL\big(p(x), q(x)\big) + \int p(z) KL\big(p(x|z), q(x|z)\big)dz      \tag{C.6} \newline
            \end{align}

            Similarly, by swapping the order of $Z$ and $X$, the following relationship can be obtaine:
            \begin{align}
                KL\big(p(x,z), q(x,z)\big) &= KL\big(p(z), q(z)\big) + \int p(x) KL\big(p(z|x), q(z|x)\big)dx       \tag{C.7}
            \end{align}

            Comparing the two equations, we can obtain:
            \begin{align}
                KL\big(p(z), q(z)\big) + \int p(x) KL\big(p(z|x), q(z|x)\big)dx =  KL\big(p(x), q(x)\big) + \int p(z) KL\big(p(x|z), q(x|z)\big)dz      \tag{C.8}
            \end{align}

            Since $q(x|z)$ and $p(x|z)$ are both transition probability densities of the Markov chain, equal to $r(x|z)$, the integral $\int p(z) KL\big(p(x|z), q(x|z)\big)dz$ equals 0. Therefore, the above equation simplifies to:
            \begin{align}
                KL\big(p(x), q(x)\big) = KL\big(p(z), q(z)\big) - \int p(x) KL\big(p(z|x), q(z|x)\big)dx            \tag{C.9}
            \end{align}

            Since KL divergence is always greater than or equal to 0, the weighted sum $\int p(x) KL\big(p(z|x), q(z|x)\big)dx$ is also greater than or equal to 0. Therefore, we can conclude:
            \begin{align}
                KL\big(p(x), q(x)\big) \le KL\big(p(z), q(z)\big)                   \tag{C.10}
            \end{align}

            </br>

            The condition for the above equation to hold is that $\int p(x) KL\big(p(z|x), q(z|x)\big)dx$ equals 0, which requires that for different conditions $x$, $p(z|x)$ and $q(z|x)$ must be equal. In most cases, when $p(z)$ and $q(z)$ are different, $p(z|x)$ and $q(z|x)$ are also different. This means that in most cases, we have
            \begin{align}
                KL\big(p(x), q(x)\big) < KL\big(p(z), q(z)\big)         \tag{C.11}
            \end{align}

            </br></br>
            <b>Corollary 2</b>

            Using Total Variance (L1 distance) as a metric, the transition transform of a Markov chain is non-expanding, which means  
            \begin{align}
                \left\lVert p(x)-q(x) \right\rVert_1\ &\le\ \left\lVert p(z) - q(z) \right\rVert_1  \tag{C.12}
            \end{align}

            Here, $p(z)$ and $q(z)$ are arbitrary probability density functions, and $r(x|z)$ is the transition probability density function of a Markov chain. We have $p(x) = \int r(x|z)p(z)dz$ and $q(x) = \int r(x|z) q(z) dz$.

            Proof：
            \begin{align}
                \left\lVert p(x)-q(x) \right\rVert_1\ &= \int \big\lvert p(x) - q(x) \big\rvert dx  \tag{C.13} \newline
                &=  \int \left\lvert \int r(x|z) p(z) dz - \int r(x|z)q(z)dz \right\rvert dx        \tag{C.14} \newline
                &=  \int \left\lvert \int r(x|z) \big(p(z)-q(z)\big) dz \right\rvert dx             \tag{C.15} \newline
                &\le \int \int r(x|z) \left\lvert \big(p(z)-q(z)\big) \right\rvert dz dx	        \tag{C.16} \newline
                &= \int \int r(x|z)dx \left\lvert \big(p(z)-q(z)\big) \right\rvert dz               \tag{C.17} \newline
                &= \int \left\lvert \big(p(z)-q(z)\big) \right\rvert dz                             \tag{C.18} \newline
                &= \left\lVert p(z) - q(z) \right\rVert_1                                           \tag{C.19}
            \end{align}

            Here, Equation C.16 applies the Absolute Value Inequality, while Equation C.18 utilizes the property of $r(x|z)$ being a probability distribution. 

            Proof completed.

            </br>

            Figure C.1 shows an example of a one-dimensional random variable, which can help better understand the derivation process described above. 
            
            The condition for the above equation to hold is that all non-zero terms inside the absolute value brackets have the same sign. As shown in Figure C.1(a), there are five absolute value brackets, each corresponding to a row, with five terms in each bracket. The above equation holds if and only if all non-zero terms in each row have the same sign. If different signs occur, this will lead to $\lVert p(x)-q(x) \rVert_1\ <\ \lVert p(z) - q(z) \rVert_1$. The number of different signs is related to the nonzero elements of the transition probability matrix. In general, the more nonzero elements there are, the more different signs there will be.

            For the posterior transform, generally, when $\alpha$ decreases (more noise), the transition probability density function will have more nonzero elements, as shown in Figure C.2(a); whereas when $\alpha$ increases (less noise), the transition probability density function will have fewer nonzero elements, as shown in Figure C.2(b).
            
            So, there is a feature: <b>when $\alpha$ decreases, it leads to $\lVert p(x)-q(x) \rVert_1$ being smaller than $\lVert p(z) - q(z) \rVert_1$, which means the shrinking rate of the posterior transform is larger.</b>
            
            <center> <img src="file/C1.png" width="1024" style="margin-top:12px"/> </center>
            <center> Figure C.1: Non-expanding under L1 norm  </center>
            </br>
            <center> <img src="file/C2.png" width="568" style="margin-top:12px"/> </center>
            <center> Figure C.2: More non-zero elements as $\alpha$ gets smaller </center>
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_non_expanding_en")

    return


def md_stationary_en():
    global g_latex_del

    title = "Appendix D Posterior Transform Converges to the Unique Stationary Distribution"
    with gr.Accordion(label=title, elem_classes="first_md", elem_id="stationary"):
        gr.Markdown(
            r"""
            According to the conclusion of Theorem 3 in <a href="#mc_basic_t3">[19]</a>, <b>an aperiodic and irreducible Markov chain will converge to a unique stationary distribution</b>. 

            The following will show that under certain conditions, the posterior transform is the transition probability density function of an <b>aperiodic and irreducible Markov chain</b>. 

            For convenience, the forward transform of the diffusion model is described below in a more general form.
            \begin{align}
                Z = \sqrt{\alpha}X + \sqrt{\beta}\ \epsilon     \tag{D.1} \newline
            \end{align}

            As described in <a href="#transform">Section 1</a>, $\sqrt{\alpha}X$ narrows the probability density function of $X$, so $\alpha$ controls the narrowing intensity, while $\beta$ controls the amount of noise added(smoothing). When $\beta = 1 - \alpha$, the above transform is consistent with the equation 1.1.
            
            The form of the posterior probability distribution corresponding to the new transformation is as follows:
            \begin{align}
                q(x|z=c)  =  \operatorname{Normalize} \Big(\  \overbrace{\frac{1}{\sqrt{2\pi}\sigma}\exp{\frac{-(x-\mu)^2}{2\sigma^2}}}^{\text{GaussFun}}\ q(x)\ \Big)  \tag{D.2} \newline
                 \text{where}\ \mu=\frac{c}{\sqrt{\alpha}}\qquad \sigma=\sqrt{\frac{\beta}{\alpha}} \qquad \text{$c$ is a fixed value}  \notag
            \end{align}

            When $\beta = 1 - \alpha$, the above transform is consistent with the equation 3.4.

            For convenience, let $g(x)$ represent GaussFun in Equation D.2.

            Since $\sqrt{\alpha}X$ narrows the probability density function $q(x)$ of $X$, this makes the analysis of the aperiodicity and irreducibility of the transition probability density function $q(x|z)$ more complex. Therefore, for the sake of analysis, we first assume $\alpha = 1$ and later analyze the case when $\alpha \neq 1$ and $\beta = 1 - \alpha$.

            <center> <img src="file/D1.png" width="960" style="margin-top:12px"/> </center>
            <center> Figure D.1: Only one component in support </center>

            <center> <img src="file/D2.png" width="960" style="margin-top:12px"/> </center>
            <center> Figure D.2: One component which can communicate with each other </center>

            </br>
            <h3 style="font-size:24px"> $\alpha=1$ </h3>

            When $\alpha=1$, if $q(x)$ and $\beta$ satisfy either of the following two conditions, the Markov chain corresponding to $q(x|z)$ is aperiodic and irreducible. 

            <ol style="list-style-type:decimal">
            <li>If the support of $q(x)$ contains only one connected component.</li>
            <li>If the support of $q(x)$ has multiple connected components, but the distance between each connected component is less than $3$ times $\sigma$. In other words, the gaps can be covered by the radius of the effective region of $g(x)$.</li>
            </ol>

            Proof：

            <ol style="list-style-type:decimal">
            <li>
            For any point $c$ in the support of $q(x)$, when $z=c$ and $x=c$, $q(x=c)>0$; from Equation D.2, we know that the center of $g(x)$ is located at $c$, so $g(x)$ is also greater than 0 at $x=c$. Therefore, according to characteristics of multiplication in the equation D.2, $q(x=c|z=c)>0$. Hence, the Markov chain corresponding to $q(x|z)$ is aperiodic. 

            For any point $c$ in the support of $q(x)$, when $z=c$, the center of $g(x)$ is located at $c$, so there exists a hypersphere with $c$ as its center ($\lVert x-c\rVert_2 < \delta$). Within this hypersphere, $q(x|z=c)>0$, which means that state $c$ can access nearby states. Since every state in the support has this property, all states within the entire support form a $\textcolor{red}{\text{Communicate Class}}$ <a href="#mc_basic_d4">[14]</a>. Therefore, the Markov chain corresponding to $q(x|z)$ is irreducible.
            
            Therefore, a Markov chain that satisfies condition 1 is aperiodic and irreducible. See the example in Figure D.1, which illustrates a single connected component 
            </li>

            <li>
            When the support set of $q(x)$ has multiple connected components, the Markov chain may have multiple communicate classes. However, if the gaps between components are smaller than $3\sigma$(standard deviation of $g(x)$), the states of each component can access each other. Thus, the Markov chain corresponding to $q(x|z)$ will have only one communicate class, similar to the case in condition 1. Therefore, a Markov chain that satisfies condition 2 is aperiodic and irreducible.
            
            In Figure D.2, an example of multiple connected components is shown.
            </li>
            </ol>

            <center> <img src="file/D3.png" width="960" style="margin-top:12px"/> </center>
            <center> Figure D.3: Two component which <b>cannot</b> communicate with each other </center>

            </br>
            <h3 style="font-size:24px"> $\alpha \neq 1$ </h3>

            When $\alpha \neq 1$, for any point $c$ within the support of $q(x)$, it follows from Equation D.2 that the center of $g(x)$ is no longer $c$ but rather $\frac{c}{\sqrt{\alpha}}$. That is to say, the center of $g(x)$ deviates from $c$, with the deviation distance being $\lVert c\rVert(\frac{1-\sqrt{\alpha}}{\sqrt{\alpha}})$. It can be observed that the larger $\lVert c\rVert$ is, the greater the deviation. See the examples in Figures D.4(c) and D.4(d) for specifics. In Figure D.4(d), when $z=2.0$, the center of $g(x)$ is noticeably offset from $x=2.0$. This phenomenon is referred to in this article as <b>the Center Deviation Phenomenon</b>.
            
            The <b>Center Deviation Phenomenon</b> will affect the properties of some states in the Markov chain.

            When the deviation distance is significantly greater than $3\sigma$, $g(x)$ may be zero at $x = c$ and its vicinity. Consequently, $q(x=c|z=c)$ may also be zero, and $q(x|z=c)$ in the vicinity of $x = c$ may also be zero. Therefore, state $c$ may not be able to access nearby states and may be periodic. This is different from the case when $\alpha=1$. Refer to the example in Figure D.5: the $\textcolor{green}{\text{green curve}}$ represents $g(x)$ for $z=6.0$, and the $\textcolor{orange}{\text{orange curve}}$ represents $q(x|z=6.0)$. Because the center of $g(x)$ deviates too much from $x=6.0$, $q(x=6.0|z=6.0)=0$.
            
            When the deviation distance is significantly less than $3\sigma$, $g(x)$ is non-zero at $x = c$ and its vicinity. Consequently, $q(x=c|z=c)$ will not be zero, and $q(x|z=c)$ in the vicinity of $x = c$ will also not be zero. Therefore, state $c$ can access nearby states and is aperiodic.
            
            Under what conditions for $c$ will the deviation distance of the center of $g(x)$ be less than $3\sigma$?
             
            \begin{align}
                \lVert c\rVert(\frac{1-\sqrt{\alpha}}{\sqrt{\alpha}})\ <\  3\frac{\sqrt{\beta}}{\sqrt{\alpha}} \qquad \Rightarrow \qquad \lVert c\rVert \ <\ 3\frac{\sqrt{\beta}}{1-\sqrt{\alpha}}      \tag{D.3} \newline
            \end{align}

            From the above, it is known that there exists an upper limit such that as long as $\lVert c\rVert$ is less than this upper limit, the deviation amount will be less than $3\sigma$. 

            When $\beta=1-\alpha$, the above expression becomes
            \begin{align}
                \lVert c\rVert \ <\ 3\frac{\sqrt{1-\alpha}}{1-\sqrt{\alpha}}    \tag{D.4} \newline
            \end{align}

            $3\frac{\sqrt{1-\alpha}}{1-\sqrt{\alpha}}$ has a strictly monotonically decreasing relationship with $\alpha$. 

            When $\alpha \in (0, 1)$，
            \begin{align}
                3\frac{\sqrt{1-\alpha}}{1-\sqrt{\alpha}} > 3        \tag{D.5} \newline
            \end{align}

            Based on the analysis above, the following conclusion can be drawn

            <ol style="list-style-type:decimal">
            <li>
            <b>If the support of $q(x)$ contains only one connected component, and the points of the support set are all within a distance less than $3\frac{\sqrt{1-\alpha}}{1-\sqrt{\alpha}}$ from the origin, then the Markov chain corresponding to $q(x|z)$ will be aperiodic and irreducible.</b>
            </li>

            <li>
            If the support of $q(x)$ contains multiple connected components, the accurate determination of whether two components can access each other becomes more complex due to the Center Deviation Phenomenon of $g(x)$. Here, we won't delve into further analysis. But just give a conservative conclusion: <b>If the points of the support are all within a distance less than $1$ from the origin, and the gaps between each connected component are all less than $2\sigma$, then the Markov chain corresponding to $q(x|z)$ will be aperiodic and irreducible.</b>
            </li>
            </ol>

            <center> <img src="file/D4.png" width="1280" style="margin-top:12px"/> </center>
            <center> Figure D.4: Center Deviation of the GaussFun </center>
            </br>
            <center> <img src="file/D5.png" width="568" style="margin-top:12px"/> </center>
            <center> Figure D.5: Deviation is More Than $3\sigma$ </center>

            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_stationary_en")

    return


def md_reference_en():
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

            <a id="mc_basic_t3" href="http://galton.uchicago.edu/~lalley/Courses/312/MarkovChains.pdf"> [19] Markov Chain:Basic Theory - Theorem 3 </a>

            <a id="mc_mt_lambda" href="https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf"> [20] Markov Chains and Mixing Times, second edition - 12.2 The Relaxation Time </a>

            <a id="non_neg_lambda" href="https://link.springer.com/book/10.1007/0-387-32792-4"> [21] Non-negative Matrices and Markov Chains - Theorem 2.10 </a>

            <a id="prml_mcmc" href="https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf"> [22] Pattern Recognition and Machine Learning - 11.2. Markov Chain Monte Carlo </a>
            
            <a id="elem" href="https://cs-114.org/wp-content/uploads/2015/01/Elements_of_Information_Theory_Elements.pdf"> [23] Elements_of_Information_Theory_Elements - 2.9 The Second Law of Thermodynamics </a>

            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_reference_en")

    return


def md_about_en():
    global g_latex_del

    with gr.Accordion(label="About", elem_classes="first_md", elem_id="about"):

        gr.Markdown(
            r"""
            <b>APP</b>: This Web APP is developed using Gradio and deployed on HuggingFace. Due to limited resources (2 cores, 16G memory), the response may be slow. For a better experience, it is recommended to clone the source code from <a href="https://github.com/blairstar/The_Art_of_DPM">github</a> and run it locally. This program only relies on Gradio, SciPy, and Matplotlib.
            
            <b>Author</b>: Zhenxin Zheng, Senior computer vision engineer with ten years of algorithm development experience, Formerly employed by Tencent and JD.com, currently focusing on image and video generation.
            
            <b>Email</b>: blair.star@163.com.
            """, latex_delimiters=g_latex_del, elem_classes="normal mds", elem_id="md_about_en")

    return


def run_app():
    
    # with gr.Blocks(css=g_css, js="() => { insert_special_formula(); }", head=js_head) as demo:
    with gr.Blocks(css=g_css, js="() => {insert_special_formula(); write_markdown();}", head=js_head) as demo:
        md_introduction_en()
        
        md_transform_en()
        
        md_likelihood_en()
        
        md_posterior_en()
        
        md_forward_process_en()
        
        md_backward_process_en()
        
        md_fit_posterior_en()
        
        md_posterior_transform_en()
        
        md_deconvolution_en()
        
        md_cond_kl_en()
        
        md_approx_gauss_en()
    
        md_non_expanding_en()
    
        md_stationary_en()
        
        md_reference_en()
        
        md_about_en()
        
    demo.launch(allowed_paths=["/"])
    
    return


if __name__ == "__main__":
    run_app()