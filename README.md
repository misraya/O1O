# O1O: Grouping Known Classes to Identify Unknown Objects as Odd-One-Out (ACCV 2024 Oral)

[Mısra Yavuz](https://scholar.google.com/citations?user=lfU8AYUAAAAJ&hl=en), [Fatma Güney](https://mysite.ku.edu.tr/fguney/)


[[`paper`](https://openaccess.thecvf.com/content/ACCV2024/papers/Yavuz_O1O_Grouping_of_Known_Classes_to_Identify_Unknown_Objects_as_ACCV_2024_paper.pdf)]
[[`supplementary`](https://openaccess.thecvf.com/content/ACCV2024/supplemental/Yavuz_O1O_Grouping_of_ACCV_2024_supplemental.pdf)]
[[`bibtex`](#cite)]
[[`arxiv`](https://arxiv.org/abs/2410.07514)]
[[`website`](https://kuis-ai.github.io/O1O/)]
[[`slides`](https://drive.google.com/file/d/1ja7PgQRmTCx6GwYnMEwU6gz_iYZZBUhd/view?usp=sharing)]

### ✨ highlights 

- 🚀 leveraging DETR extensions for faster convergence
- 🔍 first-time exploration of geometric cues for OWOD
- 💡 identifying performance loss in presence of pseudo-labels
- 🧠 shaping query representations with a novel superclass prior
- 👀 detecting unknowns the way humans identify the odd-one-out
- 🏆 achieving state-of-the-art performance on incremental OWOD benchmarks
- 🎉 O1O was selected for an **oral presentation** at [ACCV2024](https://accv2024.org/) and shortlisted as a [best paper award finalist](https://accv2024.org/awards/)! 


<div align="center">
  <img src="assets/method.png" width="100%" height="90%"/>
</div><br/>



### ⚙️ environment setup

setup a new environment by using the python and cuda versions available on your machine

```bash
conda create --name o1o python==3.12
conda activate o1o
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt 
```

download the feature extraction backbone and place it under `models/`
```bash
wget https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth
```

compile CUDA operators
```bash
cd ./models/ops
sh ./make.sh
# unit test (might fail after a point depending on your gpu memory)
python test.py
```


### 🗂️ dataset setup  

1. download coco 2017, pascal voc 2007 and 2012 datasets with annotations. 
```bash
wget http://images.cocodataset.org/zips/train2017.zip  
wget http://images.cocodataset.org/zips/val2017.zip 
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip 

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
2. unzip and untar downloaded files. 
3. run `coco2voc.py` to convert coco json annotations to xml format.
4. move all annotations under `data/OWOD/Annotations` and all images under `data/OWOD/JPEGImages`.
5. if you want to train from scratch, download and unzip our pseudo-boxes from [Google Drive](https://drive.google.com/drive/folders/1ruukm_AcvpBT0aaat7TDXzuKZTIxZkn6?usp=share_link) or generate them yourself using [GOOD's codebase](https://github.com/autonomousvision/good), and put them under `data/OWOD/pseudo-boxes`.

### 📊 evaluation

first download the pretrained models from [Google Drive](https://drive.google.com/drive/folders/15V5wrL4UwqAppbk111UQxg0l18y5i3qu?usp=share_link) and place them under `exps/o1o_pretrained_mowod/` and `exps/o1o_pretrained_sowod/`.

then you can run the following commands to evaluate all checkpoints on 4 GPUs, or use the separate evaluation scripts under `configs/pretrained/`
```
./configs/pretrained/o1o_mowod_eval_all.sh
./configs/pretrained/o1o_sowod_eval_all.sh
```

### 📈  training from scratch

to train from scratch, first ensure you have followed all the steps in the [dataset setup](#dataset-setup).

then you can run the following to execute all tasks in order on 4 GPUs, or use the separate task scripts found under `configs/from-scratch/`
```
./configs/from-scratch/o1o_mowod_all.sh
./configs/from-scratch/o1o_sowod_all.sh
```


### 📌 <a name="cite"></a> citation

if you use O1O in your research, please cite our work! 

```bibtex
@InProceedings{Yavuz_2024_ACCV,
  author    = {Yavuz, M{\i}sra and G\"uney, Fatma},
  title     = {O1O: Grouping of Known Classes to Identify Unknown Objects as Odd-One-Out},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  month     = {December},
  year      = {2024},
  pages     = {614-629}
}
```

### 🙌 acknowledgements
O1O builds on previous work [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR), [DN-DETR](https://github.com/IDEA-Research/DN-DETR), [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR), [OW-DETR](https://github.com/akshitac8/OW-DETR), [PROB](https://github.com/orrzohar/PROB/) and [GOOD](https://github.com/autonomousvision/good).  
if you found O1O useful, please consider citing these works as well!


