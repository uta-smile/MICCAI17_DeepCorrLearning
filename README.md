# MICCAI17_DeepCorrLearning
This cotain core model codes for MICCAI'17 paper

### Theano version
The work is built by theano which is somewhat out-dated. 

"DeepCorrPre.py" is used first to maximize the correlation among the views. "DeepMultiSurv.py" is used to loading saved models and transfer feature hierarchies from view commonality and specifically fine-tunes on the survival regression task. Please check the paper for more details.

### Citation
If you find this repository useful in your research, please cite:
```
@inproceedings{yao2017deep,
  title={Deep correlational learning for survival prediction from multi-modality data},
  author={Yao, Jiawen and Zhu, Xinliang and Zhu, Feiyun and Huang, Junzhou},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={406--414},
  year={2017},
  organization={Springer}
}
