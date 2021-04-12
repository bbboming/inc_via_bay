# Incremental Learning via Bayesian Optimization 

# Related Papers
* [A Novel Layer Sharing-based incremetal Learning via Bayesian Optimization, Kim, B.; Kim, T.; Choe, Y., Proceedings, 10, Nov, 2020](https://sciforum.net/manuscripts/7654/manuscript.pdf)
* [Bayesian Optimization Based Efficient Layer Sharing for Incremental Learning. Kim, B.; Kim, T.; Choe, Y. Appl. Sci. 2021, 11, 2171.](https://www.mdpi.com/2076-3417/11/5/2171)

# Datasets and Network Architectures
## Dataset
* CIFAR100 - Split into 70/30, 60/30/10
* EMNIST

## Architecture
* Resnet50
* Mobilenet-v2

## Refrence Source Code
* https://github.com/weiaicunzai/pytorch-cifar100
  
# Utilities
* make_train_test.py - Split dataset into train / test according to ratio that user is defined
* make_icifar100.py - Make incremental dataset from cifiar100
* make_subcifar.py - Make sub category dataset from cifiar100