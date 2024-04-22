# Mean Field Theory in Deep Metric Learning
Official implementation of ICLR 2024 paper "Mean Field Theory in Deep Metric Learning"

## Installation

Install [torch and torchvision compatible with your CUDA version](https://pytorch.org/) and the following libraries:

```bash
faiss-gpu
numpy
pandas
pytorch-metric-learning
tqdm
wandb
```

## Datasets

1. Download four public benchmarks for deep metric learning
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - Stanford Online Products ([Link](https://cvgl.stanford.edu/projects/lifted_struct/))
   - In-shop Clothes Retrieval ([Link](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html))

2. Extract the tgz or zip file into `./data/` (Exceptionally, for Cars-196, put the files in a `./data/cars196`)

## Training Embedding Network

### CUB-200-2011

- Train an embedding network of Inception-BN (d=512) using **MeanFieldClassWiseMultiSimilarity loss**

```bash
python train.py --gpu-id 0 \
                --loss MeanFieldClassWiseMultiSimilarity \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --alpha 0.01 \
                --beta 80 \
                --mrg 0.8 \
                --mf-reg 0 \
                --lr 1e-4 \
                --lr-ratio 2000 \
                --dataset cub \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --patience 10
```

- Train an embedding network of ResNet-50 (d=512) using **MeanFieldClassWiseMultiSimilarity loss**

```bash
python train.py --gpu-id 0 \
                --loss MeanFieldClassWiseMultiSimilarity \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --alpha 0.01 \
                --beta 80 \
                --mrg 0.8 \
                --mf-reg 0 \
                --lr 1e-4 \
                --lr-ratio 2000 \
                --dataset cub \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 5 \
                --patience 10
```

### Cars-196

- Train an embedding network of Inception-BN (d=512) using **MeanFieldClassWiseMultiSimilarity loss**

```bash
python train.py --gpu-id 0 \
                --loss MeanFieldClassWiseMultiSimilarity \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --alpha 0.01 \
                --beta 80 \
                --mrg 0.8 \
                --mf-reg 0 \
                --lr 1e-4 \
                --lr-ratio 2000 \
                --dataset cars \
                --warm 1 \
                --bn-freeze 1 \
                --lr-decay-step 20 \
                --patience 10
```

- Train an embedding network of ResNet-50 (d=512) using **MeanFieldClassWiseMultiSimilarity loss**

```bash
python train.py --gpu-id 0 \
                --loss MeanFieldClassWiseMultiSimilarity \
                --model resnet50 \
                --embedding-size 512 \
                --batch-size 120 \
                --alpha 0.01 \
                --beta 80 \
                --mrg 0.8 \
                --mf-reg 0 \
                --lr 1e-4 \
                --lr-ratio 2000 \
                --dataset cars \
                --warm 5 \
                --bn-freeze 1 \
                --lr-decay-step 10 \
                --patience 10
```

### Stanford Online Products

- Train an embedding network of Inception-BN (d=512) using **MeanFieldClassWiseMultiSimilarity loss**

```bash
python train.py --gpu-id 0 \
                --loss MeanFieldClassWiseMultiSimilarity \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --alpha 0.01 \
                --beta 80 \
                --mrg 0.8 \
                --mf-reg 0 \
                --lr 6e-4 \
                --lr-ratio 2000 \
                --dataset SOP \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --patience 10
```

### In-Shop Clothes Retrieval

- Train an embedding network of Inception-BN (d=512) using **MeanFieldClassWiseMultiSimilarity loss**

```bash
python train.py --gpu-id 0 \
                --loss MeanFieldClassWiseMultiSimilarity \
                --model bn_inception \
                --embedding-size 512 \
                --batch-size 180 \
                --alpha 0.01 \
                --beta 80 \
                --mrg 0.8 \
                --mf-reg 0 \
                --lr 6e-4 \
                --lr-ratio 2000 \
                --dataset Inshop \
                --warm 1 \
                --bn-freeze 0 \
                --lr-decay-step 20 \
                --lr-decay-gamma 0.25 \
                --patience 10
```

## Evaluating Image Retrieval

Follow the below steps to evaluate the pretrained model or your trained model.

The trained best model will be saved in the `./logs/folder_name`.

```bash
# The parameters should be changed according to the model to be evaluated.
python evaluate.py --gpu-id 0 \
                   --batch-size 120 \
                   --model bn_inception \
                   --embedding-size 512 \
                   --dataset cub \
                   --resume /set/your/model/path/best_model.pth
```

## Acknowledgments

Our code is modified and adapted on these repositories:

- [Proxy Anchor Loss for Deep Metric Learning](https://github.com/tjddus9597/Proxy-Anchor-CVPR2020)
- [PyTorch Metric learning](https://github.com/KevinMusgrave/pytorch-metric-learning)

## Citation

If you use this method or this code in your research, please cite as:


    @InProceedings{Furusawa_2024_ICLR,
      author = {Furusawa, Takuya},
      title = {Mean Field Theory for Deep Metric Learning},
      booktitle = {International Conference on Learning Representations (ICLR)},
      month = {May},
      year = {2024}
    }

