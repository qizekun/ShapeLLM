# Data

## ShapeLLM

ShapeLLM utilizes text-3D paired data provided by Cap3D for the first-stage alignment training and employs GPT-4V to construct multi-view-based general data for supervised finetuning.
To equip the model with the capability of 3D Visual Grounding, we also leverage GPT4 based on GAPartNet to build SFT data for Embodied Understanding.
All data except Cap3D_pts can be directly downloaded from [Hugging Face](https://huggingface.co/datasets/qizekun/ShapeLLM). Cap3D_pcs data needs to be obtained in pt form from the [Cap3D repository](https://huggingface.co/datasets/tiange/Cap3D/tree/main).

| Data file name                                                                                                            |        Size |
|---------------------------------------------------------------------------------------------------------------------------|------------:|
| [cap3d_objaverse_785k.json](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/cap3d_objaverse_785k.json)         |      242 MB |
| [cap3d_objaverse_sft_45k.json](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/cap3d_objaverse_sft_45k.json)   |     16.9 MB |
| [gapartnet_sft_27k_openai.json](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/gapartnet_sft_27k_openai.json) |     12.5 MB |
| [gapartnet_pcs.zip](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/gapartnet_pcs.zip)                         |     4.59 GB |
| [cap3d_pcs](https://huggingface.co/datasets/tiange/Cap3D/tree/main/PointCloud_pt_zips)                                    |    173.8 GB |
Organize the data as follows in `./playground/data/shapellm/`
```
│playground/data/shapellm/
├── cap3d_objaverse_785k.json
├── cap3d_objaverse_sft_45k.json
├── gapartnet_sft_27k_openai.json
├── gapartnet_pcs
│   ├── Box_100129_0_0.npy
│   └── ...
└── cap3d_pcs
    ├── 00000054c36d44a2a483bdbff31d8edf.pt
    └── ...
```

## ReCon++
The overall directory structure should be:
```
│llava/
│ReConV2/
│data/
├──OpenShape/
├──ModelNet/
├──ModelNetFewshot/
├──ScanObjectNN/
```

### OpenShape Dataset:

```
│OpenShape/
├──objaverse-processed/
│  └── merged_for_training_final/
│      ├── 3D-FUTURE/
│      ├── ABO/
│      ├── Objaverse/
│      └── ShapeNet/
├──meta_data/
│      ├── modelnet40/
│      ├── scanobjectnn/
│      ├── split/
│      ├── gpt4_filtering.json
│      ├── lvis_cat_name_pt_feat.npy
│      └── point_feat_knn.npy
```
Download: You can download the processed data from [OpenShape Hugging Face](https://huggingface.co/datasets/OpenShape/openshape-training-data/tree/main). Note that the rendered image data is not necessary.


### ModelNet40 Dataset: 

```
│ModelNet/
├──modelnet40_normal_resampled/
│  ├── modelnet40_shape_names.txt
│  ├── modelnet40_train.txt
│  ├── modelnet40_test.txt
│  ├── modelnet40_train_8192pts_fps.dat
│  └── modelnet40_test_8192pts_fps.dat
```
Download: You can download the processed data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md), or download from the [official website](https://modelnet.cs.princeton.edu/#) and process it by yourself.

### ModelNet Few-shot Dataset:
```
│ModelNetFewshot/
├──5way10shot/
│  ├── 0.pkl
│  ├── ...
│  ├── 9.pkl
├──5way20shot/
│  ├── ...
├──10way10shot/
│  ├── ...
├──10way20shot/
│  ├── ...
```

Download: Please download the data from [Point-BERT repo](https://github.com/lulutang0608/Point-BERT/blob/49e2c7407d351ce8fe65764bbddd5d9c0e0a4c52/DATASET.md). We use the same data split as theirs.

### ScanObjectNN Dataset:
```
│ScanObjectNN/
├──main_split/
│  ├── training_objectdataset_augmentedrot_scale75.h5
│  ├── test_objectdataset_augmentedrot_scale75.h5
│  ├── training_objectdataset.h5
│  ├── test_objectdataset.h5
├──main_split_nobg/
│  ├── training_objectdataset.h5
│  └── test_objectdataset.h5
```
Download: Please download the data from the [official website](https://hkust-vgd.github.io/scanobjectnn/).
