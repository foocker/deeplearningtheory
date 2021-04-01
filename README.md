# Deep Learning Theory
整理了一些深度学习的理论相关内容，持续更新。
## Overview
1. [Recent advances in deep learning theory](https://arxiv.org/pdf/2012.10931.pdf)
总结了目前深度学习理论研究的六个方向的一些结果，概述型，没做深入探讨(2021)。

    + 1.1 complexity and capacity-basedapproaches for analyzing the generalizability of deep learning; 

    + 1.2 stochastic differential equations andtheir dynamic systems for modelling stochastic gradient descent and its variants, which characterizethe optimization and generalization of deep learning, partially inspired by Bayesian inference; 

    + 1.3 thegeometrical structures of the loss landscape that drives the trajectories of the dynamic systems;

    + 1.4 theroles of over-parameterization of deep neural networks from both positive and negative perspectives; 

    + 1.5 theoretical foundations of several special structures in network architectures; 

    + 1.6 the increasinglyintensive concerns in ethics and security and their relationships with generalizability

## Course
2. [Theory of Deep Learning](https://www.ideal.northwestern.edu/special-quarters/fall-2020/)TTIC,西北大学等组织的一系列课程和讲座，基础课程涉及DL的基础(符号化，简化后的数学问题和结论)，信息论和学习，统计和计算，信息论，统计学习和强化学习(2020)。

3. [MathsDL-spring19](https://joanbruna.github.io/MathsDL-spring19/),MathDL系列，18,19,20年均有。

    + 3.1  Geometry of Data

        + Euclidean Geometry: transportation metrics, CNNs , scattering.
        + Non-Euclidean Geometry: Graph Neural Networks.
        + Unsupervised Learning under Geometric Priors (Implicit vs explicit models, microcanonical, transportation metrics).
        + Applications and Open Problems: adversarial examples, graph inference, inverse problems.

    + 3.2 Geometry of Optimization and Generalization

        + Stochastic Optimization (Robbins & Munro, Convergence of SGD)
        + Stochastic Differential Equations (Fokker-Plank, Gradient Flow, Langevin + + Dynamics, links with SGD; open problems)
        Dynamics of Neural Network Optimization (Mean Field Models using Optimal Transport, Kernel Methods)
        + Landscape of Deep Learning Optimization (Tensor/Matrix factorization, Deep Nets; open problems).
        + Generalization in Deep Learning.

    + 3.3  Open qustions on Reinforcement Learning

## Architecture
5. [Partial Differential Equations is All You Need for Generating Neural Architectures -- A Theory for Physical Artificial Intelligence Systems](https://arxiv.org/abs/2103.08313) 将统计物理的反应扩散方程，量子力学中的薛定谔方程，傍轴光学中的亥姆霍兹方程统一整合到神经网络偏微分方程中(NPDE)，利用有限元方法找到数值解，从离散过程中，构造了多层感知，卷积网络，和循环网络，并提供了优化方法L-BFGS等，主要是建立了经典物理模型和经典神经网络的联系(2021)。

## Approximation
6. NN Approximation Theory
    + 6.0 [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem)NN逼近从这里开始(1991)
    + 6.1 [Cybenko’s Theorem and the capabilityof a neural networkas function approximator](https://www.mathematik.uni-wuerzburg.de/fileadmin/10040900/2019/Seminar__Artificial_Neural_Network__24_9__.pdf)一二维shallow神经网络的可视化证明(2019)
    + 6.2 [Depth Separation for Neural Networks](https://arxiv.org/abs/1702.08489) 三层神经网络的表示能力比两层有优越性的简化证明 (2017)
    + 6.3 [赵拓等三篇](https://www.zhihu.com/question/347654789/answer/1480974642)(2019)
    + 6.4 [Neural Networks with Small Weights and Depth-Separation Barriers](https://arxiv.org/abs/2006.00625) Gal Vardi等证明了对某些类型的神经网络, 用k层的多项式规模网络需要任意weight, 但用3k+3层的多项式规模网络只需要多项式大小的 weight(2020)。

## Optimization
7. SGD
    + 7.1 [nesterov-accelerated-gradient](https://paperswithcode.com/method/nesterov-accelerated-gradient)

8. [offconvex](http://www.offconvex.org/)几个学术工作者维护的AI博客。
    + 8.1 [beyondNTK](http://www.offconvex.org/2021/03/25/beyondNTK/) 什么时候NN强于NTK？
    + 8.2 [instahide](http://www.offconvex.org/2020/11/11/instahide/) 如何在不泄露数据的情况下优化模型？
    + 8.3 [implicit regularization in DL explained by norms?](http://www.offconvex.org/2020/11/27/reg_dl_not_norm/)

## Geometry
9. Optima transmission
    + 9.1 [深度学习的几何学解释](http://www.engineering.org.cn/ch/10.1016/j.eng.2019.09.010)(2020)

## Others
4. [Theoretical issues in deep networks](https://www.pnas.org/content/117/48/30039) 表明指数型损失函数中存在隐式的正则化，其优化的结果和一般损失函数优化结果一致，优化收敛结果和梯度流的迹有关，目前还不能证明哪个结果最优(2020)。
