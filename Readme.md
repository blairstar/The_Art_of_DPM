# The Art of DPM

An in-depth analysis about diffusion probabilily model

It is written in Chinese. English version will be released next month.



#### 本文的核心特色:

1. 理论上分析出数据分布$q(x)$向噪声分布$q(z_T)$转换的规律。

2. 理论上证明了"优化重建项($\iint q(x)q(z_1|x)\log p(x|z_1)dxdz_1$)的本质是使$p(x|z_1)$去拟合$q(x|z_1)$"。

3. 理论上证明了"优化一致项($\iint q(z_t,x)KL[q(z_{t-1}|z_t,x)\Vert p(z_{t-1}|z_t)]dz_tdx$)的本质是使$p(z_{t-1}|z_t)$去拟合$q(z_{t-1}|z_t)$"。

4. 实验证明了"$q(z_{t-1}|z_t)$逆变换具备良好的抗噪性能"。

5. 实验证明了"$q(z_{t-1}|z_t)$逆变换是一个压缩映射(Contraction Mapping)"。

6. 实验证明了"$q(z_{t-1}|z_t)$压缩映射具有一个不动点(Fixed Point)，并且是$q(z_{t-1})$"。


## 目录

1. 基础准备  
   - 1.1 随机变量的期望  
   - 1.2 随机变量的变换的期望  
   - 1.3 通过随机变量替换计算变换的期望  
   - 1.4 一些简单变换的期望和方差  
   - 1.5 随机变量的变换的概率分布  
   - 1.6 两个独立随机变量之和的概率分布  
   - 1.7 高斯分布的性质  
   - 1.8 log函数的性质  
   - 1.9 蒙特卡洛积分(Monte Carlo Integral)  
   - 1.10 概率密度函数的性质  
   - 1.11 贝叶斯公式(Bayes Formula)和全概率公式(Total Probability Formula)  
   - 1.12 KL散度及其"条件上限"  
2. 隐变量模型的复杂度及采样方法  
	- 2.1 更复杂的概率分布  
	- 2.2 并不困难的采样  
3. 三种隐变量模型及相应的参数学习方法  
	- 3.1 Mixture Model及EM算法  
	- 3.2 VAE模型及其学习方法  
	- 3.3 DPM模型及其学习方法    
		+ 3.3.1 DPM模型的形式及其Lower Bound    
		+ 3.3.2 q概率模型和Lower Bound的简化  
		+ 3.3.3 三种优化(预测)方式  
		+ 3.3.4 优化MLE  
	- 3.4 DPM模型的进一步分析  
		+ 3.4.1 概括重要的结论  
		+ 3.4.2 进一步理解目标函数  
		+ 3.4.3 理解噪声分布向数据分布转变的过程  
		+ 3.4.4 q(z<sub>t-1</sub>|z<sub>t</sub>)概率分布的特点  
		+ 3.4.5 q(z<sub>t-1</sub>|z<sub>t</sub>)逆变换的输入敏感度  
		+ 3.4.6 p(z<sub>t-1</sub>|z<sub>t</sub>)拟合误差对逆变换的影响  
		+ 3.4.7 压缩映射q(z<sub>t-1</sub>|z<sub>t</sub>)的不动点  
		+ 3.4.8 DPM模型设计要点  
		+ 3.4.9 DPM模型的独特之处  
		+ 3.4.10 是否可通过“逆卷积”恢复q(x)  
	- 3.5 融合DPM模型和VAE模型  