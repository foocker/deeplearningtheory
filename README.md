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
    + 6.5 [Universality of Deep Convolutional Neural Networks](https://arxiv.org/pdf/1805.10769.pdf)卷积网络的通用逼近能力，及其核心要素，Ding-Xuan Zhou(2018)，20年发表。

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

## Book
10. [Theory of Deep Learning(draft)](https://www.cs.princeton.edu/courses/archive/fall19/cos597B/lecnotes/bookdraft.pdf)Rong Ge 等(2019)。

11. [Spectral Learning on Matrices and Tensors](https://arxiv.org/pdf/2004.07984.pdf)Majid Janzamin等(2020)

19. [Deep Learning Architectures A Mathematical Approach](https://www.springer.com/gp/book/9783030367206)(2020),你可以libgen获取，内容如其名字,大概包含：工业问题，DL基础(激活，结构，优化等),函数逼近，万有逼近，RELU等逼近新研究，函数表示，以及两大方向，信息角度，几何角度等相关知识，实际场景中的卷积，池化，循环，生成，随机网络等具体实用内容的数学化，另外附录集合论，测度论，概率论，泛函，实分析等基础知识。
20. [The Principles of Deep Learning Theory](https://arxiv.org/pdf/2106.10165.pdf)(2021)Daniel A. Roberts and Sho Yaida(mit)，Beginning from a first-principles component-level picture of networks，本书解释了如何通过求解层到层迭代方程和非线性学习动力学来确定训练网络输出的准确描述。一个主要的结果是网络的预测是由近高斯分布描述的，网络的深度与宽度的纵横比控制着与无限宽度高斯描述的偏差。本书解释了这些有效深度网络如何从训练中学习非平凡的表示，并更广泛地分析非线性模型的表示学习机制。从近内核方法的角度来看，发现这些模型的预测对底层学习算法的依赖可以用一种简单而通用的方式来表达。为了获得这些结果，作者开发了表示组流（RG 流）的概念来表征信号通过网络的传播。通过将网络调整到临界状态，他们为梯度爆炸和消失问题提供了一个实用的解决方案。作者进一步解释了 RG 流如何导致近乎普遍的行为，从而可以将由不同激活函数构建的网络做类别划分。Altogether, they show that the depth-to-width ratio governs the effective model complexity of the ensemble of trained networks。利用信息理论，作者估计了模型性能最好的最佳深宽比，并证明了残差连接能将深度推向任意深度。利用以上理论工具，就可以更加细致的研究架构的归纳偏差，超参数，优化。
21. [Physics-based Deep Learning](https://arxiv.org/pdf/2109.05237.pdf)(2021)N. Thuerey, P. Holl,etc.[github resources](https://github.com/thunil/Physics-Based-Deep-Learning)深度学习与物理学的联系。比如基于物理的损失函数，可微流体模拟，逆问题的求解，Navier-Stokes方程的前向模拟，Controlling Burgers’ Equation和强化学习的关系等。

## Session
21. [Foundations of Deep Learning](https://simons.berkeley.edu/programs/dl2019)(2019)，西蒙研究中心会议。
22. [Deep Learning Theory 4](https://icml.cc/virtual/2021/session/12048)(2021, ICML)Claire Monteleoni主持...,深度学习理论会议4，包含论文和视频。
23. [Deep Learning Theory 5 ](https://icml.cc/virtual/2021/session/12057)(2021,ICML)MaYi主持...，深度学习理论会议5，包含论文和视频。

## generalization
1. [Robust Learning with Jacobian Regularization](https://arxiv.org/abs/1908.02729)(2019)Judy Hoffman..., 


## Others
4. [Theoretical issues in deep networks](https://www.pnas.org/content/117/48/30039) 表明指数型损失函数中存在隐式的正则化，其优化的结果和一般损失函数优化结果一致，优化收敛结果和梯度流的迹有关，目前还不能证明哪个结果最优(2020)。
12. [The Dawning of a New Erain Applied Mathematics](https://www.ams.org/journals/notices/202104/rnoti-p565.pdf)Weinan E关于在DL的新处境下结合历史的工作范式给出的指导性总结(2021)。
13. [Mathematics of deep learning from Newton Institute](https://www.newton.ac.uk/event/mdl)。
14. [DEEP NETWORKS FROM THE PRINCIPLE OF RATE REDUCTION](https://openreview.net/forum?id=G70Z8ds32C9)，白盒神经网络。
15. [redunet_paper](https://github.com/ryanchankh/redunet_paper)白盒神经网络代码。
16. [Theory of Deep Convolutional Neural Networks:Downsampling](https://www.cityu.edu.hk/rcms/pdf/XDZhou/dxZhou2020b.pdf)下采样的数学分析Ding-Xuan Zhou(2020)
17. [Theory of deep convolutional neural networks II: Spherical analysis](https://arxiv.org/abs/2007.14285)还有III：radial functions 逼近，(2020)。不过这些工作到底如何，只是用数学转换了一下，理论上没做过多贡献，或者和实际结合没难么紧密，还不得而知。
18. [The Modern Mathematics of Deep Learning](https://arxiv.org/abs/2105.04026)(2021)主要是deep laerning的数学分析描述，涉及的问题包括：超参数网络的通用能力，深度在深度模型中的核心作用，深度学习对维度灾难的克服，优化在非凸优化问题的成功，学习的表示特征的数学分析，为何深度模型在物理问题上有超常表现，模型架构中的哪些因素以何种方式影响不同任务的学习中的不同方面。
19. [Topos and Stacks of Deep Neural Networks](https://arxiv.org/abs/2106.14587)(2021)每一个已知的深度神经网络(DNN)对应于一个典型的 Grothendieck 的 topos 中的一个对象; 它的学习动态对应于这个 topos 中的一个态射流。层中的不变性结构(如 CNNs 或 LSTMs)与Giraud's stacks相对应。这种不变性被认为是泛化性质的原因，即从约束条件下的学习数据进行推断。纤维代表前语义类别(Culioli，Thom) ，其上人工语言的定义，内部逻辑，直觉主义，经典或线性(Girard)。网络的语义功能是用这种语言表达理论的能力，用于回答输入数据输出中的问题。语义信息的量和空间的定义与香农熵的同源解释相类似。他们推广了 Carnap 和 Bar-Hillel (1952)所发现的度量。令人惊讶的是，上述语义结构被分类为几何纤维对象在一个封闭的Quillen模型范畴，然后他们引起同时局部不变的 dnn 和他们的语义功能。Intentional type theories (Martin-Loef)组织这些对象和它们之间的纤维化。信息内容和交换由 Grothendieck's derivators分析。
20. [Visualizing the Emergence of Intermediate Visual Patterns in DNNs](https://arxiv.org/abs/2111.03505)(2021,NIPS)文章设计了一种神经网络中层特征的可视化方法，使得能
（1）更直观地分析神经网络中层特征的表达能力，并且展示中层特征表达能力的时空涌现；
（2）量化神经网络中层知识点，从而定量地分析神经网络中层特征的质量；
（3）为一些深度学习技术（如对抗攻击、知识蒸馏）提供新见解。
21.  [神经网络的博弈交互解释性](https://zhuanlan.zhihu.com/p/264871522/)(知乎)。上交大张拳石团队研究论文整理而得，作为博弈交互解释性的体系框架（不怎么稳固）。
22.  [Advancing mathematics by guiding human intuition with AI](https://www.nature.com/articles/s41586-021-04086-x)(2021,nature)机器学习和数学家工作的一个有机结合，主要利用机器学习分析众多特征和目标变量的主要相关因子，加强数学家的直觉，该论文得到了两个漂亮的定理，一个拓扑，一个表示论。可参考[回答](https://www.zhihu.com/question/503185412/answer/2256015652)。