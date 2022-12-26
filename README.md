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
2. 🔥 [On the Principles of Parsimony and Self-Consistency for the Emergence of Intelligence](https://arxiv.org/abs/2207.04630)(2022.7.马毅、沈向洋、曹颖)"任何一个智能系统，作为一个看似简单的自主闭环：信息、控制、对策、优化、神经网络，紧密结合，缺一不可。总而言之，人工智能的研究从现在开始，应该能够也必须与科学、数学、和计算紧密结合。从最根本、最基础的第一性原理（简约、自洽）出发，把基于经验的归纳方法与基于基础原理的演绎方法严格地、系统地结合起来发展。理论与实践紧密结合、相辅相成、共同推进我们对智能的理解。" 知乎解读可看：https://zhuanlan.zhihu.com/p/543041107 。

## Course
2. [Theory of Deep Learning](https://www.ideal.northwestern.edu/special-quarters/fall-2020/)TTIC,西北大学等组织的一系列课程和讲座，基础课程涉及DL的基础(符号化，简化后的数学问题和结论)，信息论和学习，统计和计算，信息论，统计学习和强化学习(2020)。

2+. [2021 Deep learning theory lecture note](https://mjt.cs.illinois.edu/dlt/ )
    + 2.1 逼近，优化，通用性，三方面做了总结，核心内容网页可见，比较友好。

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
4. [IFT 6169: Theoretical principles for deep learning](http://mitliagkas.github.io/ift6085-dl-theory-class/)(2022 Winter),大多内容较为基础，传统。
    + 4.1 拟定课题
        + Generalization: theoretical analysis and practical bounds
        + Information theory and its applications in ML (information bottleneck, lower bounds etc.)
        +  Generative models beyond the pretty pictures: a tool for traversing the data manifold, projections, completion, substitutions etc.
        + Taming adversarial objectives: Wasserstein GANs, regularization approaches and + controlling the dynamics
        + The expressive power of deep networks (deep information propagation, mean-field analysis of random networks etc.)
5. [深度学习几何课程](https://geometricdeeplearning.com/lectures/)(2022, Michael Bronstein)内容比较高级.
    + 5.1 2022 年的 GDL100 共包含 12 节常规课程、3 节辅导课程和 5 次专题研讨。12 节常规课程主要介绍了几何深度学习的基本概念知识，包括高维学习、几何先验知识、图与集合、网格（grid）、群、测地线（geodesic）、流形（manifold）、规范（gauge）等。3 节辅导课主要面向表达型图神经网络、群等变神经网络和几何图神经网络。

    + 5 次专题研讨的话题分别是：
        1. 从多粒子动力学和梯度流的角度分析神经网络；
        2. 表达能力更强的 GNN 子图；
        3. 机器学习中的等变性；
        4. 神经 sheaf 扩散：从拓扑的角度分析 GNN 中的异质性和过度平滑；
        5. 使用 AlphaFold 进行高度准确的蛋白质结构预测。

6. [Advanced Topics in Machine Learning and Game Theory](https://feifang.info/advanced-topics-in-machine-learning-and-game-theory-fall-2022/)游戏，强化方面的课程，2022。
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

9. [Adam](https://arxiv.org/abs/1412.6980)
    + 9.1 [deep-learning-dynamics-paper-list](https://github.com/zeke-xie/deep-learning-dynamics-paper-list)关于DL优化动力学方面研究的资料收集。
    + 9.2 [Adai](https://github.com/zeke-xie/adaptive-inertia-adai) Adam的优化版本Adai，Adam逃离鞍点很快，但是不能像SGD一样擅长找到flat minima。作者设计一类新的自适应优化器Adai结合SGD和Adam的优点。Adai逃离鞍点速度接近Adam,寻找flat minima能接近SGD。其知乎介绍可看[Adai-zhihu]((https://www.zhihu.com/question/323747423/answer/2576604040))


## Geometry
9. Optima transmission
    + 9.1 [深度学习的几何学解释](http://www.engineering.org.cn/ch/10.1016/j.eng.2019.09.010)(2020)

## Book
10. [Theory of Deep Learning(draft)](https://www.cs.princeton.edu/courses/archive/fall19/cos597B/lecnotes/bookdraft.pdf)Rong Ge 等(2019)。

11. [Spectral Learning on Matrices and Tensors](https://arxiv.org/pdf/2004.07984.pdf)Majid Janzamin等(2020)

19. [Deep Learning Architectures A Mathematical Approach](https://www.springer.com/gp/book/9783030367206)(2020),你可以libgen获取，内容如其名字,大概包含：工业问题，DL基础(激活，结构，优化等),函数逼近，万有逼近，RELU等逼近新研究，函数表示，以及两大方向，信息角度，几何角度等相关知识，实际场景中的卷积，池化，循环，生成，随机网络等具体实用内容的数学化，另外附录集合论，测度论，概率论，泛函，实分析等基础知识。
20. [The Principles of Deep Learning Theory](https://arxiv.org/pdf/2106.10165.pdf)(2021)Daniel A. Roberts and Sho Yaida(mit)，Beginning from a first-principles component-level picture of networks，本书解释了如何通过求解层到层迭代方程和非线性学习动力学来确定训练网络输出的准确描述。一个主要的结果是网络的预测是由近高斯分布描述的，网络的深度与宽度的纵横比控制着与无限宽度高斯描述的偏差。本书解释了这些有效深度网络如何从训练中学习非平凡的表示，并更广泛地分析非线性模型的表示学习机制。从近内核方法的角度来看，发现这些模型的预测对底层学习算法的依赖可以用一种简单而通用的方式来表达。为了获得这些结果，作者开发了表示组流（RG 流）的概念来表征信号通过网络的传播。通过将网络调整到临界状态，他们为梯度爆炸和消失问题提供了一个实用的解决方案。作者进一步解释了 RG 流如何导致近乎普遍的行为，从而可以将由不同激活函数构建的网络做类别划分。Altogether, they show that the depth-to-width ratio governs the effective model complexity of the ensemble of trained networks。利用信息理论，作者估计了模型性能最好的最佳深宽比，并证明了残差连接能将深度推向任意深度。利用以上理论工具，就可以更加细致的研究架构的归纳偏差，超参数，优化。[原作者的视频说明](https://www.youtube.com/watch?v=wXZKoHEzASg)(2021.12.1)
21. [Physics-based Deep Learning](https://arxiv.org/pdf/2109.05237.pdf)(2021)N. Thuerey, P. Holl,etc.[github resources](https://github.com/thunil/Physics-Based-Deep-Learning)深度学习与物理学的联系。比如基于物理的损失函数，可微流体模拟，逆问题的求解，Navier-Stokes方程的前向模拟，Controlling Burgers’ Equation和强化学习的关系等。
22. [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478)(Michael M. Bronstein, Joan Bruna, Taco Cohen, Petar Veličković,2021),见上面课程5:深度学习几何课程.

## Session
21. [Foundations of Deep Learning](https://simons.berkeley.edu/programs/dl2019)(2019)，西蒙研究中心会议。
22. [Deep Learning Theory 4](https://icml.cc/virtual/2021/session/12048)(2021, ICML)Claire Monteleoni主持...,深度学习理论会议4，包含论文和视频。
23. [Deep Learning Theory 5 ](https://icml.cc/virtual/2021/session/12057)(2021,ICML)MaYi主持...，深度学习理论会议5，包含论文和视频。

## generalization
1. [Robust Learning with Jacobian Regularization](https://arxiv.org/abs/1908.02729)(2019)Judy Hoffman..., 
2. [Predicting Generalization using GANs](http://www.offconvex.org/2022/06/06/PGDL/)(2022.6),用GAN来评估泛化性.
3. [Implicit Regularization in Tensor Factorization: Can Tensor Rank Shed Light on Generalization in Deep Learning?](http://www.offconvex.org/2021/07/08/imp-reg-tf/)(2021.7)Tensor Rank 能否揭示深度学习中的泛化？
4. [如何通过Meta Learning实现域泛化Domain Generalization](https://mp.weixin.qq.com/s/o1liWf9B4_LntBeyV2bVOg)(2022.4),[Domain Generalization CVPR2022](https://mp.weixin.qq.com/s/HkjHEqs8d85VPdgpaHEzPQ)博文参考.
5. [Generalization-Causality](https://github.com/yfzhang114/Generalization-Causality) 一博士关于domain generalization等工作的实时汇总。
6. [Implicit Regularization in Hierarchical Tensor Factorization and Deep Convolutional Networks](http://www.offconvex.org/2022/07/15/imp-reg-htf-cnn/)(Noam Razin  •  Jul 15, 2022)

    + 6.1 Across three different neural network types (equivalent to matrix, tensor, and hierarchical tensor factorizations), we have an architecture-dependant notion of rank that is implicitly lowered. Moreover, the underlying mechanism for this implicit regularization is identical in all cases. This leads us to believe that implicit regularization towards low rank may be a general phenomenon. If true, finding notions of rank lowered for different architectures can facilitate an understanding of generalization in deep learning.

    + 6.2 Our findings imply that the tendency of modern convolutional networks towards locality may largely be due to implicit regularization, and not an inherent limitation of expressive power as often believed. More broadly, they showcase that deep learning architectures considered suboptimal for certain tasks can be greatly improved through a right choice of explicit regularization. Theoretical understanding of implicit regularization may be key to discovering such regularizers.


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
23. 🔥[A New Perspective of Entropy](https://math3ma.institute/wp-content/uploads/2022/02/bradley_spring22.pdf)(2022) 通过莱布尼兹微分法则(Leibniz rule)将信息熵,抽象代数,拓
扑学联系起来。该文章是一个零基础可阅读的综述,具体参考[Entropy as a Topological Operad Derivation ](https://www.mdpi.com/1099-4300/23/9/1195)(2021.7,Tai-Danae Bradley.)
24. [minerva](https://storage.googleapis.com/minerva-paper/minerva_paper.pdf)(2022)google提出的解题模型,在公共高等数学等考试中比人类平均分高.[测试地址](https://minerva-demo.
github.io/#category=Algebra&index=1).
25. 🔥[An automatic theorem proving project](https://gowers.wordpress.com/2022/04/28/announcing-an-automatic-theorem-proving-project/#more-6531)菲尔兹获得者数学家高尔斯关于
自动证明数学定理的项目进展[How can it be feasible to find proofs?](https://drive.google.com/file/d/1-FFa6nMVg18m1zPtoAQrFalwpx2YaGK4/view)(2022, W.T. Gowers).
26. [GRAND: Graph Neural Diffusion ](https://papertalk.org/papertalks/32188)(2021)该网站包含了一些相似论文资料,[项目地址graph-neural-pde](https://github.com/twitter-research
/graph-neural-pde),其优化版本[GRAND++](https://openreview.net/forum?id=EMxu-dzvJk).(2022).有博文介绍[图神经网络的困境，用微分几何和代数拓扑解决](https://mp.weixin.qq.com/s/CFNvgn6vaYcI36QJNa3_dw)仅供参
考.
27. [Weinan È-A Mathematical Perspective on Machine Learning](https://opade.digital/)(2022.icm),room1最后一排,鄂维南在icm的演讲视频.
28.  [contrastive learning](https://zhuanlan.zhihu.com/p/524733769)证明包括InfoNCE在内的一大类对比学习目标函数，等价于一个有两类变量（或者说两类玩家）参与的交替优化（或者说游戏）过程.
<<<<<<< HEAD
29.  [可解释性：Batch Normalization未必客观地反应损失函数信息](https://zhuanlan.zhihu.com/p/523627298)2022,张拳石等.
30. [Homotopy Theoretic and Categorical Models of Neural Information Networks](https://arxiv.org/abs/2006.15136)该工作第一作者俄罗斯数学家Yuri Manin，2020工作，2022年8月arxiv有更新。[ncatlab有讨论](https://nforum.ncatlab.org/discussion/13133/understanding-preprint-topos-and-stacks-of-deep-neural-networks/)。
31. [Deep learning via dynamical systems: An approximation perspective](https://ems.press/journals/jems/articles/5404458)动力系统逼近。[论文见](https://cpb-us-w2.wpmucdn.com/blog.nus.edu.sg/dist/d/11132/files/2021/01/main-jems.pdf)。
32. [群论角度](https://www.youtube.com/playlist?list=PL8FnQMH2k7jzPrxqdYufoiYVHim8PyZWd)群论角度去理解的一系列视频，群论视角，2014年出现过，视频系统讲解，2022年。
=======
29.  [可解释性：Batch Normalization未必客观地反应损失函数信息](https://zhuanlan.zhihu.com/p/523627298)2022,张拳石等.BN操作使得神经网络训练无法客观反应损失函数的信息。具体地，本文理论证明了当我们对深度神经网络的损失函数进行泰勒展开时， BN操作阻塞了损失函数的一阶项和二阶项的反向传播。在实验中，作者发现BN操作有时候阻碍了神经网络对特定特征的学习。论文的结论是很清晰的,具体看论文.
30.  [Why Adversarial Training of ReLU Networks is Difficult?](https://arxiv.org/abs/2205.15130)2022.5,Xu Cheng等
     1. 我们推导了对抗扰动的解析解，该解析解揭示了对抗扰动的动力学本质。
     2. 基于对抗扰动的解析解，我们理论解释了对抗训练优化困难的原因。
     3. 基于上述理论，我们统一分析十篇前人对对抗训练的研究的内在机理。
     4. 对抗扰动会增强输入x的Hessian矩阵最大特征值对应特征向量方向上的梯度分量，并且当对抗力度较大时，对抗扰动只会沿着少数特征值较大的方向变化。
     5. 对抗训练可以被认为增加了Hessian矩阵对网络参数更新的影响，这使得对抗训练相较于正常训练，更容易出现网络参数在某一方向振荡的情况，增加了对抗训练的困难程度。
     6. 等
31. [Trap of Feature Diversity in the Learning of MLPs](https://arxiv.org/abs/2112.00980)2022.7,Dongrui Liu等.
    1.  我们发现了一个长期被忽略的基础但反直觉的现象，即在多层感知机训练的前期（在很早期Loss不下降的那一段），特征多样性会快速下降，甚至不同类别的不同样本学习到几乎一样的特征。这时神经网络有些像一个自激系统。这种现象是十分反常的，而且它会破坏多层感知机的训练。
    2.  当训练任务比较难的时候，很多神经网络往往“训练不动”，理论上恰恰属于卡在了这个神经网络的系统自激阶段。
    3.  我们从学习动力学 (learning dynamics) 的角度理论上解释了这一现象，并且基于理论分析，我们解释了四种可以缓解这个现象操作的原因。
    4.  [原作者解析](https://zhuanlan.zhihu.com/p/521453526)



>>>>>>> dcbd248fa1d60b14e14dbff7edc6f0ec963f48c8
 ## DeepModeling
 [DeepModeling](https://deepmodeling.com/)鄂维南等组织,一种新的研究范式,将DL建模渗透到科研中,这里会开源很多对新或旧问题的DL建模方案.[其github地址](https://github.com/deepmode
ling).空了看情况解析某些工作.
 
 ## 数学形式主义与计算机
1. [The Future of Mathematics？ ](https://www.bilibili.com/video/av71583469)(2019) Kevin Buzzard就lean的一场讲座，评论区有对应讲义资料。
2. [数学形式主义的兴起](https://mp.weixin.qq.com/s/-XosE3LzA8wfFv-38EIfKQ)(2022.7)Kevin Buzzard教授在2022本届国际数学家大会一小时报告演讲中提供了一些信息和思考见解。讲述了数学
形式主义与人工智能、机器学习和开源社区的共同努力，用计算机做奥数题、检查数学证明过程是否有误、甚至自动发现和形式化证明数学定理，在理论和实践中又会碰撞出什么火花，又会如何囿于...
3. [专访ICM 2022国际数学家大会一小时报告者Kevin Buzzard：计算机可以成为数学家吗？——译自量子杂志](https://mp.weixin.qq.com/s/VWuRyxkl0xgZWcqRn0WJAw)比较好的采访,值得看看.数学家让计算机科学家了解到数学很难,这个部分,在被逐渐理解,且计算机系统检查,可能会解决这个难点.还有那些炫酷的项目,球面外翻,费马大定理,非常值得关注.
4. [Deep Maths-machine learning and mathematics](https://www.youtube.com/watch?v=wbJQTtjlM_w),重新发现Euler多面体公式 （对之前工作的细节的更进一步说明）,涉及组合不变量猜想，庞加莱猜想，瑟斯顿几何猜想，扭结图等（涉及的面很大，但都是一带而过）。![image](./imgs/ex1_knot.png)
## Discussion
1. [怎样看待Ali Rahimi 获得 NIPS 2017 Test-of-time Award后的演讲？](https://www.zhihu.com/question/263711574)17年就有人(张心欣,王刚等)指出了DL的缺陷,和这个领域中人的特点,过去5年了,还是那样.不过如23 能看出,meta的做应用的田渊栋还在坚守理论.
2. [深度学习领域有哪些瓶颈？](https://www.zhihu.com/question/40577663/answer/2593884415)张拳石新的吐槽,以及最新成果汇集.
