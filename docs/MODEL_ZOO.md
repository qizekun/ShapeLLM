# Model Zoo
## ShapeLLM
All the checkpoints can be found in [Hugging Face](https://huggingface.co/collections/qizekun/shapellm-65e978379c1260a85abe8aee).

| Model    | Size | sft data                                                                                              | Checkpoint                                                                                        |
|----------|------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| ShapeLLM | 7B   | [objaverse](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/cap3d_objaverse_sft_45k.json)  | [qizekun/ShapeLLM_7B_general_v1.0](https://huggingface.co/qizekun/ShapeLLM_7B_general_v1.0)       |
| ShapeLLM | 13B  | [objaverse](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/cap3d_objaverse_sft_45k.json)  | [qizekun/ShapeLLM_13B_general_v1.0](https://huggingface.co/qizekun/ShapeLLM_13B_general_v1.0)     |
| ShapeLLM | 7B   | [gapartnet](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/gapartnet_sft_27k_openai.json) | [qizekun/ShapeLLM_7B_gapartnet_v1.0](https://huggingface.co/qizekun/ShapeLLM_7B_gapartnet_v1.0)   |
| ShapeLLM | 13B  | [gapartnet](https://huggingface.co/datasets/qizekun/ShapeLLM/blob/main/gapartnet_sft_27k_openai.json) | [qizekun/ShapeLLM_13B_gapartnet_v1.0](https://huggingface.co/qizekun/ShapeLLM_13B_gapartnet_v1.0) |


## ReCon++

| Model     | Size    | Type                         | Acc.  | Checkpoint                                                                                     |
|-----------|---------|------------------------------|-------|------------------------------------------------------------------------------------------------|
| ReCon++   | Base    | Pretrain (10k)               | N.A.  | [best_lvis.pth]()                                                                              |
| ReCon++   | Large   | Pretrain (10k)               | N.A.  | [best_lvis.pth](https://huggingface.co/qizekun/ReConV2/blob/main/zeroshot/large/best_lvis.pth) |
| ReCon++   | Base    | Zeroshot on Objaverse-LVIS   | 53.2  | [best_lvis.pth]()                                                                              |
| ReCon++   | Base    | Zeroshot on ModelNet40       | 86.5  | [best_modelnet40_overall.pth]()                                                                |
| ReCon++   | Base    | Zeroshot on ScanObjectNN     | 63.6  | [best_scanobjectnn.pth]()                                                                      |
| ReCon++   | Large   | Zeroshot on Objaverse-LVIS   | 53.7  | [best_lvis.pth](https://huggingface.co/qizekun/ReConV2/blob/main/zeroshot/large/best_lvis.pth) |
| ReCon++   | Large   | Zeroshot on ModelNet40       | 87.3  | [best_modelnet40_overall.pth]()                                                                |
| ReCon++   | Large   | Zeroshot on ScanObjectNN     | 65.4  | [best_scanobjectnn.pth]()                                                                      |
| --------- | ------- | ---------------------------- |       | ------------                                                                                   |
| ReCon++   | Base    | Pretrain (1k)                | N.A.  |                                                                                                |
| ReCon++   | Large   | Pretrain (1k)                | N.A.  |                                                                                                |
| ReCon++   | Base    | finetune on OBJ_BG           | 98.62 |                                                                                                |
| ReCon++   | Base    | finetune on OBJ_ONLY         | 96.21 |                                                                                                |
| ReCon++   | Base    | finetune on PB_T50_RS        | 93.34 |                                                                                                |
| ReCon++   | Base    | finetune on ModelNet40 (1k)  | 94.6  |                                                                                                |
| ReCon++   | Base    | finetune on ModelNet40 (8k)  | 94.8  |                                                                                                |
| ReCon++   | Large   | finetune on OBJ_BG           | 98.80 |                                                                                                |
| ReCon++   | Large   | finetune on OBJ_ONLY         | 97.59 |                                                                                                |
| ReCon++   | Large   | finetune on PB_T50_RS        | 95.25 |                                                                                                |
| ReCon++   | Large   | finetune on ModelNet40 (1k)  | 94.8  |                                                                                                |
| ReCon++   | Large   | finetune on ModelNet40 (8k)  | 95.0  |                                                                                                |