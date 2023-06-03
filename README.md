# GenMesh
<img src="./demo/demo.gif" data-canonical-src="./demo/demo.gif" width="1200" height="700" />
The is the released codes for "Single-view 3D Mesh Reconstruction for Seen and Unseen Categories". Published in IEEE Transactions on Image Processing (TIP).

[Paper](https://ieeexplore.ieee.org/document/10138738) -
[Arxiv](https://arxiv.org/abs/2303.04341) -

## Requirements
The codes have been tested on the linux Ubuntu 20.04 system with NVIDIA RTX3090ti. The enviroment inlcude:
* Python=3.9
* Pytorch=1.12.1
* Torchvision=0.13.1
* CUDA=11.6
  
Please clone the repository and navigate into it in your terminal, its location is assumed for all subsequent commands.

## Installation
The `nvf.yml` file contains all necessary python dependencies for the project. You can create the anaconda environment using: 
```
conda env create -f nvf.yml
conda activate nvf
```
There are other dependencies to be installed.

1. To train the model, relavattive operations in [point-transfomrer](https://github.com/POSTECH-CVLab/point-transformer) are needed. 
2. To extract meshes, [meshUDF](https://github.com/cvlab-epfl/MeshUDF) is needed. 
3. To test the model, libaries for [chamfer distance](https://github.com/otaheri/chamfer_distance), [earth mover distance](https://github.com/daerduoCarey/PyTorchEMD) are needed. 

The commands have been incorperated by `create_env.sh`. You can install them via runing the script:
```
bash create_env.sh
```
Or you can install step by step by yourself.

## Data Preparation
First, create a configuration file in folder `configs/`, use `configs/shapenet_cube_offset_generalization_pt_vq_k16.txt` as reference and see configs/config_loader.py for detailed explanation of all configuration options. Change the desired data directory with variable `SAVE_DIR` in `create_split_generalization.py`, `convert_to_scaled_off.py ` and `boundary_sampling.py`. Please also make sure to move the split files `split_generalization_*.npz` from `dataprocessing` to your desired `SAVE_DIR`.

Next, prepare the data for NVF using
```
python dataprocessing/preprocess.py --config configs/shapenet_cube_offset_generalization_pt_vq_k16.txt
```
You can generate a random test/training/validation split of the data using
```
python dataprocessing/create_split.py --config configs/shapenet_cube_offset_generalization_pt_vq_k16.txt
```
but replacing `configs/shapenet_cube_offset_generalization_pt_vq_k16.txt` in the commands with the desired configuration.

## Training
To train your NVF, you can change the parameters in the configs and run:
```
python train_generalization.py --config ./configs/${exp_name}.txt 2>&1|tee ${save_dir}/log.txt
```
In the `experiments/ `folder you can find an experiment folder containing the model checkpoints, the checkpoint of validation minimum, and a folder containing a tensorboard summary, which can be started at with
```
tensorboard --logdir experiments/${exp_name}/summary/ --host 0.0.0.0
```
## Generation
To generate meshes after training:
```
python generation.py
```
Please specify the desired model before running.
## Test
To test results after generation:
```
python test.py
```
Please specify the desired experiment name before running.

## Contact
For questions and comments please leave your questions in the issue or contact Xianghui Yang via email xianghui.yang@sydney.edu.au.

## Acknowledge
The code is modified from the [NDF](https://github.com/jchibane/ndf). Thanks for open-sourcing.

```
@ARTICLE{10138738,
  author={Yang, Xianghui and Lin, Guosheng and Zhou, Luping},
  journal={IEEE Transactions on Image Processing}, 
  title={Single-view 3D Mesh Reconstruction for Seen and Unseen Categories}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TIP.2023.3279661}}

```