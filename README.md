# ShapeLLM: Universal 3D Object Understanding for Embodied Interaction

*We present ShapeLLM, the first 3D Multimodal Large Language Model designed for embodied interaction, exploring a universal 3D object understanding with 3D point clouds and languages.*

[Zekun Qi](https://qizekun.github.io/), [Runpei Dong](https://runpeidong.com/), [Shaochen Zhang](), [Haoran Geng](https://geng-haoran.github.io/), [Chunrui Han](), [Zheng Ge](https://joker316701882.github.io/), [Li Yi](https://ericyi.github.io) and [Kaisheng Ma](http://group.iiis.tsinghua.edu.cn/~maks/leader.html)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shapellm-universal-3d-object-understanding/zero-shot-transfer-3d-point-cloud-2)](https://paperswithcode.com/sota/zero-shot-transfer-3d-point-cloud-2?p=shapellm-universal-3d-object-understanding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shapellm-universal-3d-object-understanding/zero-shot-3d-classification-on-objaverse-lvis)](https://paperswithcode.com/sota/zero-shot-3d-classification-on-objaverse-lvis?p=shapellm-universal-3d-object-understanding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shapellm-universal-3d-object-understanding/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=shapellm-universal-3d-object-understanding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shapellm-universal-3d-object-understanding/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=shapellm-universal-3d-object-understanding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shapellm-universal-3d-object-understanding/few-shot-3d-point-cloud-classification-on-3)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-3?p=shapellm-universal-3d-object-understanding)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/shapellm-universal-3d-object-understanding/3d-point-cloud-linear-classification-on)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on?p=shapellm-universal-3d-object-understanding)

<a href="https://qizekun.github.io/shapellm/"><img src="https://img.shields.io/badge/Project-Page-Green"></a>
<a href="https://arxiv.org/abs/2402.17766"><img src="https://img.shields.io/badge/Paper-PDF-orange"></a> 

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)

**Usage and License Notices**: The data and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

<div style="text-align: center;">
    <img src="assets/framework.jpg" width=100% >
</div>

Codes and checkpoints will be released in one week.


## Citation

If you find ShapeLLM or ReCon++ useful for your research and applications, please cite using this BibTeX:
```bibtex

@article{qi2024shapellm,
  author = {Qi, Zekun and Dong, Runpei and Zhang, Shaochen and Geng, Haoran and Han, Chunrui and Ge, Zheng and Yi, Li and Ma, Kaisheng},
  title = {ShapeLLM: Universal 3D Object Understanding for Embodied Interaction},
  journal = {arXiv preprint arXiv:2402.17766},
  year = {2024},
}

@inproceedings{qi2023recon,
  title={Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining},
  author={Qi, Zekun and Dong, Runpei and Fan, Guofan and Ge, Zheng and Zhang, Xiangyu and Ma, Kaisheng and Yi, Li},
  booktitle={International Conference on Machine Learning (ICML) },
  year={2023}
}
```

## Acknowledgement

This codebase is built upon [LLaVA](https://github.com/haotian-liu/LLaVA), [OpenShape](https://github.com/Colin97/OpenShape_code), [ReCon](https://github.com/qizekun/ReCon) and [PointGPT](https://github.com/CGuangyan-BIT/PointGPT).

## Related Works

- [Point-Bind & Point-LLM](https://arxiv.org/abs/2309.00615)
- [3D-LLM](https://arxiv.org/abs/2307.12981)
- [PointLLM](http://arxiv.org/abs/2308.16911)
