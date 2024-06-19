# Attribution Priors applied to ECG data from the PTB Database

This project contains my personal research into using Expected Gradients and Integrated Gradients embedded in CNNs with PTB Diagnostic ECG Database

My own BSc (Hons) Degree dissertation focused on ECG data extracted from the [MIT-BIH ECG data](https://www.physionet.org/content/mitdb/1.0.0/), to train and test novel deep learning models incorporating attribution priors into their training steps.

These novel models are modified Convolutional Neural Networks, and the corresponding attribution priors derive from _Expected Gradients_ proposed by [Erion et al.](https://arxiv.org/abs/1906.10670) and _Integrated Gradients_ proposed by [Sundarajan et al.](https://arxiv.org/abs/1703.01365).

I hope to expand upon my previous work by evaluating these novel models' performances on the much larger [PTB Diagnostic ECG Database](https://physionet.org/content/ptbdb/1.0.0/). Equally this project aims to produce better explainability results and provide more valuable insight to the models processes.

### Repositories providing ground work

-   https://github.com/suinleelab/attributionpriors
-   https://github.com/hiranumn/IntegratedGradients
-   https://github.com/tensorflow/docs/blob/master/site/en/tutorials/interpretability/integrated_gradients.ipynb
