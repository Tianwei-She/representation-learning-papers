# Representation Learning Papers
May include papers on representation learning, unsupervised learning on images or videos, metric learning and other interesting topics.

## Image Representation Learning
- **Instance Recognition** Unsupervised Feature Learning via Non-Parametric Instance Discrimination, CVPR 2018 [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/0801.pdf)
  - ![instance-recognition](figures/instance-recognition.png)
  - **Training**: an instance-level classification task, where the "label" can be regarded as the feature vector for each image/instance in the memory bank. 
  - Non-parametric Softmax
    - <img src="figures/IR-non-parametric-softmax.png" alt="IR-non-parametric-softmax" width="300"/>
    - Generalizable to new classes/instances, Computationally efficient by eliminating weights
  - Noise-Contrastive Estimation (I don't fully understand it for now)
  - **Testing**: weighted kNN or SVM trained on learned feature vectors
  - Experiments
    - On CIFAR10, the non-parametric softmax performs ~15% better than parametric softmax; NCE is an efficient approximation.
    - Image classification on ImageNet
    - Scene classification on Places, using the feature extraction networks trained on ImageNet without finetuning
    - Semi-supervised learning
    - Object detection

- **MoCo** Momentum Contrast for Unsupervised Visual Representation Learning [[pdf]]( https://arxiv.org/pdf/1911.05722.pdf )

  - *Contrastive Learning* = To train an encoder for a *dictionary look-up* task/to build a discrete dictionary on high-dimensional continuous inputs such as images; *Contrastive Loss* = A function whose value is low when an encoded query $q$ is similar to its positive key $k_+$ and dissimilar to all other negative keys.
  - *InfoNCE* and the *instance discrimination task* is used in this paper :  
    <img src="figures/infoNCE.png" alt="infoNCE" width="200"/>

  - Previous mechanisms of constrastive losses  
    <img src="figures/contrastive_loss_mechanisms.png" alt="contrastive_loss_mechanisms" width="700"/>
    - *End-to-end*  update uses samples in the current mini-batch as the dictionary -> the dictionary size is coupled with the mini-batch size
    - *Memory bank* can support a large dictionary size, but it keeps keys encoded at multiple different steps which are less consistent
  - Momentum Contrast
    - Dictionary as a queue (current mini-batch is enqueued and the oldest mini-batch is dequeued) -> the dictionary size is decoupled from the mini-batch size, so that a large dictionary is possible
    - Momentum update (only the parameters of query encoder are updated by back-propagation) -> the key encoder evolves smoothly, so the keys in the queues are encoded by relatively similar encoders 
      <img src="figures/moco_momentum_update.png" alt="momentum_update" width="250"/>



Deep Learning vs. Traditional Computer Vision https://arxiv.org/ftp/arxiv/papers/1910/1910.13796.pdf 

DeepInfoMax - Learning deep representations by mutual information estimation and maximization (Yoshua Bengio)   https://arxiv.org/pdf/1808.06670.pdf 

InfoNCE - Representation learning with contrastive predictive coding. arXiv:1807.03748, 2018 

CMC (Contrastive Multiview Coding)  http://people.csail.mit.edu/yonglong/yonglong/cmc_icml_workshop.pdf 

CPC (Contrastive Predictive Coding)  https://arxiv.org/pdf/1807.03748.pdf

Revisiting Self-Supervised Visual Representation Learning https://arxiv.org/pdf/1901.09005.pdf

Unsupervised Embedding Learning via Invariant and Spreading Instance Feature  https://arxiv.org/pdf/1904.03436.pdf 

Learning Representations by Maximizing Mutual Information Across Views  https://arxiv.org/pdf/1906.00910.pdf 

Data-Efficient Image Recognition with Contrastive Predictive Coding  https://arxiv.org/pdf/1905.09272.pdf 

BigBiGAN - Large Scale Adversarial Representation Learning https://arxiv.org/pdf/1907.02544.pdf

Scaling and Benchmarking Self-Supervised Visual Representation Learning, ICCV 2019 https://arxiv.org/pdf/1905.01235.pdf

DeeperCluster - Unsupervised Pre-Training of Image Features on Non-Curated Data https://arxiv.org/pdf/1905.01278.pdf

On Mutual Information Maximization for Representation Learning (A survey paper on image representation learning using MI maximization) https://arxiv.org/pdf/1907.13625.pdf

PIRL - Self-Supervised Learning of Pretext-Invariant Representations https://arxiv.org/pdf/1912.01991.pdf

VIVI - Self-Supervised Learning of Video-Induced Visual Invariances https://arxiv.org/pdf/1912.02783.pdf

SimCLR - A Simple Framework for Contrastive Learning of Visual Representations https://arxiv.org/pdf/2002.05709.pdf

MoCo vs. SimCLR - Improved Baselines with Momentum Contrastive Learning https://arxiv.org/abs/2003.04297


## Video Representation Learning
Shuffle and learn: unsupervised learning using temporal order verification, ECCV 2016 https://arxiv.org/pdf/1603.08561.pdf

Self-supervised video representation learning with odd-one-out networks, CVPR 2017 https://arxiv.org/pdf/1611.06646.pdf

OPN - Unsupervised Representation Learning by Sorting Sequences, ICCV 2017 http://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Unsupervised_Representation_Learning_ICCV_2017_paper.pdf

Geometry Guided Convolutional Neural Networks for Self-Supervised Video Representation Learning, CVPR 2018 http://openaccess.thecvf.com/content_cvpr_2018/papers/Gan_Geometry_Guided_Convolutional_CVPR_2018_paper.pdf

Self-Supervised Video Representation Learning with Space-Time Cubic Puzzles https://arxiv.org/pdf/1811.09795.pdf

Self-supervised Spatio-temporal Representation Learning for Videos by Predicting Motion and Appearance Statistics, CVPR 2019 https://arxiv.org/pdf/1904.03597.pdf

Self-supervised Spatiotemporal Learning via Video Clip Order Prediction, CVPR 2019 http://openaccess.thecvf.com/content_CVPR_2019/papers/Xu_Self-Supervised_Spatiotemporal_Learning_via_Video_Clip_Order_Prediction_CVPR_2019_paper.pdf

DistInit: Learning Video Representations Without a Single Labeled Video, ICCV 2019 https://arxiv.org/pdf/1901.09244.pdf

RotNet - Self-Supervised Spatiotemporal Feature Learning via Video Rotation Prediction https://arxiv.org/pdf/1811.11387.pdf

Learning Video Representations Using Contrastive Bidirectional Transformer https://arxiv.org/pdf/1906.05743.pdf

Video Representation Learning by Dense Predictive Coding https://arxiv.org/pdf/1909.04656.pdf

End-to-End Learning of Visual Representations from Uncurated Instructional Videos https://arxiv.org/pdf/1912.06430.pdf

Cooperative learning of audio and video models from self-supervised synchronization, NIPS 2018 https://arxiv.org/pdf/1807.00230.pdf


## Video Datasets
Video Dataset Overview, by Antoine Miech https://www.di.ens.fr/~miech/datasetviz/

Recent evolution of modern datasets for human activity recognition: a deep survey https://link.springer.com/content/pdf/10.1007/s00530-019-00635-7.pdf
