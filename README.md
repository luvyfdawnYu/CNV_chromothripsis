# CNV chromothripsis
This is the source code to replicate GECNVNet, proposed in the paper: Deep Graph Learning-based Chromothripsis Detection and Multiple Myeloma Outcome Prediction Using Copy Number Variation.

The dataset used in the paper is also accompanied.

The prerequisites are Python 3.9 with arm architecture support, PyTorch 1.9.0, and PyG 2.1.0. All codes are tested on a Macbook Pro laptop with a M1 Pro CPU and 32GB memory (not tested on x86-based Windows/Linux/Unix platform).

To replicate the GECNVNet, please run the main.py file.

For example:

`python main.py --split 1 --test_seed 2000 --batch_size 64`

There are some adjustable (hyper)parameters that you can modify in the GECNVNet training.

### For training:

--split: in the range of [1, 5]. It decides which split to run._(mandatory)_

--test_seed: integer. It decides the 10-fold cross validation split seed and the PyTorch initialization seed._(mandatory)_

--batch_size: integer. It decides the size of a batch used in training._(mandatory)_

--learning_rate: float. It decides the learning rate used in the training.

--num_epochs: integer. It decides the training epochs in every cross-validation training process.

--osbfb_seed: integer. It decides the oversampling random seed.

### For modifying GECNVNet architecture:
--GE_dim: 2 integers, e.g., `16 32`. It decides the output demension of each Graph Transformer Layer.

--LF_stride: integer. It decides how many fully-connected layers are contained in on locally feature extraction layer.

--LF_input_shape: 4 integers, e.g., `1024 768 512 256`. It decides the input dimension of each locally feature extraction layer in the WHOLE locally feature extraction module.

--LF_output_shape: 4 integers, e.g., `24 16 8 4`. It decides the output dimension of each fully-connected layer in the WHOLE locally feature extraction module. It must be the result of LF_input_shape divided by LF_stride.

--NL_input_shape: integer. It decides the input vector dimension of the nonlinear feature interaction module.

--NL_output_shape: integer. It decides the output vector dimension of the nonlinear feature interaction module.

### For customized loss function:
--gamma: float. It decides the $\gamma$ used in the customized loss.

--epsilon: float. It decides the $\epsilon$ usde in the customized loss.

