# Learning Multiview 3D Point Cloud Registration 
This repository provides code and data to train and evaluate the LMPCR, the first end-to-end algorithm for multiview registration of raw point clouds in a globally consistent manner. It represents the official implementation of the paper:

### [Learning Multiview 3D Point Cloud Registration (CVPR 2020).](https://arxiv.org/pdf/2001.05119.pdf)
\*[Zan Gojcic](https://www.ethz.ch/content/specialinterest/baug/institute-igp/geosensors-and-engineering-geodesy/en/people/scientific-assistance/zan-gojcic.html),\* [Caifa Zhou](https://ch.linkedin.com/in/caifa-zhou-7a461510b), [Jan D. Wegner](http://www.prs.igp.ethz.ch/content/specialinterest/baug/institute-igp/photogrammetry-and-remote-sensing/en/group/people/person-detail.html?persid=186562), [Leonidas J. Guibas](https://geometry.stanford.edu/member/guibas/), [Tolga Birdal](http://tbirdal.me/)\
|[EcoVision Lab ETH Zurich](https://prs.igp.ethz.ch/ecovision.html) | [Guibas Lab Stanford University](https://geometry.stanford.edu/index.html)|\
\* Equal contribution

We present a novel, end-to-end learnable, multiview 3D point cloud registration algorithm. Registration of multiple scans typically follows a two-stage pipeline: the initial pairwise alignment and the globally consistent refinement. The former is often ambiguous due to the low overlap of
neighboring point clouds, symmetries and repetitive scene parts. Therefore, the latter global refinement aims at establishing the cyclic consistency across multiple scans and helps in resolving the ambiguous cases. In this paper we propose, to the best of our knowledge, the first end-to-end algorithm for joint learning of both parts of this two-stage problem. Experimental evaluation on well accepted benchmark datasets shows that our approach outperforms the state-of-the-art by a significant margin, while being end-to-end trainable and computationally less costly. Moreover, we present detailed analysis and an ablation study that validate
the novel components of our approach. 

![LM3DPCR](figures/LM3DPCR.jpg?raw=true)

### Citation

If you find this code useful for your work or use it in your project, please consider citing:

```shell
@inproceedings{gojcic2020LearningMultiview,
	title={Learning Multiview 3D Point Cloud Registration},
	author={Gojcic, Zan and Zhou, Caifa and Wegner, Jan D and Guibas, Leonidas J and Birdal, Tolga},
	booktitle={International conference on computer vision and pattern recognition (CVPR)},
	year={2020}
}
```

### Contact
If you have any questions or find any bugs, please let us know: Zan Gojcic {firstname.lastname@geod.baug.ethz.ch}

## Current state of the repository
Currently the repository contains only part of the code connected to the above mentioned publication and will be consistently updated in the course of the following days. The whole code will be available the following weeks. 

**NOTE**: The published model is not the same as the model used in the CVPR paper. The results can therefore slightly differ from the ones in the paper. The models will be updated in the following days (with the fully converged ones).

## Instructions
The code was tested on Ubuntu 18.04 with Python 3.6, pytorch 1.5, CUDA 10.1.243, and GCC 7.

### Requirements
After cloning this repository, you can start by creating a virtual environment and installing the requirements by running:

```bash
conda create --name lmpr python=3.6
source activate lmpr
conda config --append channels conda-forge
conda install --file requirements.txt
conda install -c open3d-admin open3d=0.9.0.0
conda install -c intel scikit-learn
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

Our network uses [FCGF](https://github.com/chrischoy/FCGF) feature descriptor which is based on the [MinkowskiEnginge](https://github.com/StanfordVL/MinkowskiEngine) library for sparse tensors. In order to install Minkowski Engine run:

```bash
source activate lmpr
git clone https://github.com/StanfordVL/MinkowskiEngine.git
cd MinkowskiEngine
conda install numpy mkl-include
export CXX=g++-7; python setup.py install
cd ../
```

Finally, our network supports furthest point sampling, when sampling the interest points. To this end we require the [PointNet++](https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2/models) library that can be installed as follows: 

```bash
source activate lmpr
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
cd Pointnet2_PyTorch
pip install -r requirements.txt
cd ../
```

## Pretrained models
We provide the pretrained models for [FCGF]((https://github.com/chrischoy/FCGF)) feature descriptor, our pairwise registration block, and jointly trained pairwise registration model. They can be downloaded using:

 ```bash
bash scripts/download_pretrained_models.sh
```

## Datasets
### Pairwise registration
In order to train the pairwise registration model, you have to download the full [3DMatch](http://3dmatch.cs.princeton.edu/) dataset. To (pre)-train the registration blocks you can either download the preprocessed dataset (~160GB) using

```bash
bash scripts/download_3DMatch_train.sh preprocessed
```

This dataset contains the pointwise correspondences established in the [FCGF](https://github.com/chrischoy/FCGF) feature space for all point cloud pairs of 3DMatch dataset.

We also provide the raw 3DMatch data (~4.5GB) and a script to generate the preprocess training data `./scripts/extract_data.py` that can be used either with the raw 3DMatch data or your personal dataset. The raw dataset can be downloaded using

```bash
bash scripts/download_3DMatch_train.sh raw
```

And then processed (extract FCGF feature descriptors, establish the correspondences, and save training data) using:

```bash
source activate lmpr
python ./scripts/extract_data.py \
			--source_path ./data/train_data/ \
			--target_path ./data/train_data/ \
			--dataset 3d_match \
			--model ./pretrained/fcgf/model_best.pth \
			--voxel_size 0.025 \
			--n_correspondences 20000 \
			--inlier_threshold 0.05 \
			--extract_features \
			--extract_correspondences \
			--extract_precomputed_training_data \
			--with_cuda \

bash scripts/download_preprocessed_3DMatch.sh
```
## Training
Current release supports only training the pairwise registration network. The repository will be further updated in the next weeks. In order to train the pairwise registration network from scratch using the precomputed data run 
```bash
source activate lmpr
python train.py ./configs/pairwise_registration/OANet.yaml

```
The training parameters can be set in `./configs/pairwise_registration/OANet.yaml`. 

In order to fine tune the pairwise registration network in an end-to-end manner (including the FCGF block), the raw data has to be used. The code to sample the batches will be released in the next weeks.

## Evaluation 
We provide the scripts for the automatic evaluation of our method on the 3DMatch and Redwood dataset. The results for our method can differ slightly from the results in the CVPR paper as we have retrained the model. Due to the different implementation the results for RANSAC might also differ slightly from
the results of the [official evaluation script](https://github.com/andyzeng/3dmatch-toolbox/tree/master/evaluation/geometric-registration).

### 3DMatch
To evaluate on 3DMatch you can either download the raw evaluation data (~450MB) using
```bash
bash scripts/download_3DMatch_eval.sh raw
```
or the processed data together with the results for our method and RANSAC (~2.7GB) using 
```bash
bash scripts/download_3DMatch_eval.sh preprocessed
```

If you download the raw data you first have to process it (extract features and correspondences)

```bash
source activate lmpr
python ./scripts/extract_data.py \
			--source_path ./data/eval_data/ \
			--target_path ./data/eval_data/ \
			--dataset 3d_match \
			--model ./pretrained/fcgf/model_best.pth \
			--voxel_size 0.025 \
			--n_correspondences 5000 \
			--extract_features \
			--extract_correspondences \
			--with_cuda \

```

Then you can run the pairwise registration evaluation using our RegBlock (with all points) as

```bash
source activate lmpr
python ./scripts/benchmark_pairwise_registration.py \
		--source ./data/eval_data/ \
		--dataset 3d_match \
		--method RegBlock \
		--model ./pretrained/RegBlock/model_best.pt \
		--only_gt_overlap \
```

This script assumes that the correspondences were already estimated using `./scripts/extract_data.py` and only benchmarks the registration algorithms. To improve efficiency the registration parameters will only be computed for the ground truth overlapping pairs, when the flag `--only_gt_overlap` is set. This does not change the registration recall, which is used as the primary evaluation metric. In order to run the estimation on all n choose 2 pairs simply omit the `--only_gt_overlap` (results in ~10 times longer computation time on 3DMatch dataset)


### Redwood
To evaluate the generalization performance of our method on redwood you can again either download the raw evaluation data (~1.3GB) using
```bash
bash scripts/download_redwood_eval.sh raw
```
or the processed data together with the results for our method and RANSAC (~2.9GB) using 
```bash
bash scripts/download_redwood_eval.sh preprocessed
```

The rest of the evaluation follow the same procedure as for 3DMatch, you simply have to replace the dataset argument with `--dataset redwood`

**NOTE**: Using the currently provided model the performance is slightly worse then reported in the paper. The model will be updated in the following days.

### Demo 

Pairwise registration demo can be run after downloading the pretrained models as 
```bash
source activate lmpr
python ./scripts/pairwise_demo.py \
		./configs/pairwise_registration/demo/config.yaml \
		--source_pc ./data/demo/pairwise/raw_data/cloud_bin_0.ply \
		--target_pc ./data/demo/pairwise/raw_data/cloud_bin_1.ply \
		--model pairwise_reg.pt \
		--verbose \
		--visualize
```
which will register the two point clouds, visualize them before and after the registration, and save the estimated transformation parameters.