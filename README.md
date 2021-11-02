# HERALD: An Annotation Efficient Method to Train User Engagement Predictors in Dialogs

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Weixin-Liang/HERALD/blob/master/LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2106.00162-b31b1b.svg)](https://arxiv.org/abs/2106.00162)
[![Python 3.8](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Pytorch](https://img.shields.io/badge/Pytorch-1.8-red.svg)](https://shields.io/)


This repo provides the PyTorch source code of our paper: 
[HERALD: An Annotation Efficient Method to Train User Engagement Predictors in Dialogs](https://aclanthology.org/2021.acl-long.283/) (ACL 2021). 
[[PDF]](https://arxiv.org/abs/2106.00162)
[[Video: Prof. Zhou Yu]](https://drive.google.com/file/d/1xjNUrkmSm-eBEQLBR2UXl2YpXkvz54-2/view?usp=sharing)






```
@inproceedings{liang2021herald,
  author =  {Weixin Liang and Kai-Hui Liang and Zhou Yu},
  title =   {{HERALD:} An Annotation Efficient Method to Train User Engagement Predictors in Dialogs},
  year =    {2021},  
  booktitle = {{ACL}},
  publisher = {Association for Computational Linguistics}
}
```



## Abstract
*Open-domain dialog systems have a user-centric goal: to provide humans with an engaging conversation experience. User engagement is one of the most important metrics for evaluating open-domain dialog systems, and could also be used as real-time feedback to benefit dialog policy learning. Existing work on detecting user disengagement typically requires hand-labeling many dialog samples. 
We propose HERALD, an annotation efficient framework that reframes the training data annotation process as a denoising problem.
Specifically, instead of manually labeling training samples, we first use a set of labeling heuristics to automatically label training samples. We then denoise the weakly labeled data using Shapley algorithm. Finally, we use the denoised data to train a user engagement detector. 
Our experiments show that HERALD improves annotation efficiency significantly and 
achieves 86\% user disengagement detection accuracy in two dialog corpora.
Our implementation is available at https://github.com/Weixin-Liang/HERALD/*


<p align='center'>
  <img width='100%' src='./figures/engagement_flow.png' />
</p>


## Stage 1: Auto-label Training Data with Heuristic Functions
<p align='center'>
  <img width='100%' src='./figures/heuristics.png' />
  <em>Table: Our labeling heuristics designed to capture user disengagement in dialogs.  A dialog turn is considereddisengaged if any of the heuristic rules applies to the user responses. </em>
</p>


## Stage 2: Denoise with Shapley Algorithm & Fine-tune

### Dependencies

Run the following commands to create a conda environment (assuming CUDA10.1):
```bash
conda create -n herald python=3.6
conda activate herald
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install matplotlib scipy
conda install -c conda-forge scikit-learn 
conda install -c conda-forge transformers
conda install pandas
```
Please check [shapley/requirements.txt](shapley/requirements.txt) or [shapley/requirements.yml](shapley/requirements.yml) for additional details about the dependencies (Note that you don't need to install all of them). 

### BERT-based Dialog Classifier
Please check [shapley/bert_dialog_engagement_classifier.py](shapley/bert_dialog_engagement_classifier.py) and [shapley/data_utils.py](shapley/data_utils.py). The code is built upon the github repo [ABSA-PyTorch](https://github.com/songyouwei/ABSA-PyTorch). 
Many thanks to the authors and developers!

#### Training
```sh
python bert_dialog_engagement_classifier.py --model_name bert_spc
```
#### Running with Custom Dialog Dataset
Please check [shapley/convai_data/convai_dataloader.py](shapley/convai_data/convai_dataloader.py) for supporting custom dialog dataset. 

### Running the Data Shapley Algorithm
Shapley algorithm computes a Shapley value for each training datum, which quantifies the contribution of each training datum to the prediction and performance of a deep network. Low Shapley value data capture outliers and corruptions. 
Therefore, we can identify and denoise the incorrectly-labeled data by computing their Shapley values, and then fine-tune the model on cleaned training set. 

To obtain a closed-form solution of Shapley value, we extract the features of training data points and apply a K-nearest-neighbour classifier. The Shapley value of each training point can be calculated recursively as follows: 
<p align='center'>
  <img width='50%' src='./figures/shapley.png' />
</p>

Please check [shapley/shapley.py](shapley/shapley.py) for the implementation of the shapley algorithm. Note that you need to first extract the features for training datapoints before running the K-nearest-neighbour based Shapley algorithm. 
In particular, the [core function](shapley/shapley.py#L63-L72) for calculating the single point data shapley value is: 
```python
def single_point_shapley(xt_query, y_tdev_label):
    distance1 = np.sum(np.square(X-xt_query), axis=1)
    alpha = np.argsort(distance1)
    shapley_arr = np.zeros(N)
    for i in range(N-1, -1, -1): 
        if i == N-1:
            shapley_arr[alpha[i]] = int(y[alpha[i]] == y_tdev_label) /N
        else:
            shapley_arr[alpha[i]] = shapley_arr[alpha[i+1]] + \
              ( int(y[alpha[i]]==y_tdev_label) - int(y[alpha[i+1]]==y_tdev_label) )/K * min(K,i+1)/(i+1)
    return shapley_arr
```
Here we use (i+1) since i starts from zero in our python implementaion. 



## Related Papers on Data Shapley
[Beyond User Self-Reported Likert Scale Ratings: A Comparison Model for Automatic Dialog Evaluation](https://www.aclweb.org/anthology/2020.acl-main.126/) (ACL 2020). 
Weixin Liang, James Zou and Zhou Yu. 
[[PDF]](https://www.aclweb.org/anthology/2020.acl-main.126.pdf)
[[Video]](https://slideslive.com/38928690/beyond-user-selfreported-likert-scale-ratings-a-comparison-model-for-automatic-dialog-evaluation)
[[Stanford AI Lab Blog]](https://ai.stanford.edu/blog/acl-2020/)
[[Slides]](https://drive.google.com/file/d/1hpuUCyz81bqtg1-De9C1ostgiJA519Vj/view?usp=sharing)
[[Code]](https://github.com/Weixin-Liang/dialog_evaluation_CMADE/)


[Data Shapley: Equitable Data Valuation for Machine Learning. ](https://arxiv.org/abs/1904.02868) (ICML 2019). Amirata Ghorbani, James Zou. 
[[PDF]](http://proceedings.mlr.press/v97/ghorbani19c/ghorbani19c.pdf)
[[Video]](https://slideslive.com/38917630/supervised-learning)
[[Poster]](https://drive.google.com/file/d/19iX2faH2Y0SE5Yn_yCmOaKAUvp57oB6y/view)
[[Slides]](https://docs.google.com/presentation/d/10Crejw-CgyS0G_16KC8cHcnoVPdDsibfFmmiHbN22Ek/edit)
[[Code]](https://github.com/amiratag/DataShapley)

## Contact
Thank you for your interest in our work! Please contact us at kl3312@columbia.edu, wxliang@stanford.edu for any questions, comments, or suggestions! 
