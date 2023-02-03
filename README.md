<div align="center">

# SegFormer-inference-template

  <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
  <a href="https://huggingface.co/"><img alt="Hugging Face" src="https://img.shields.io/badge/-huggingface-yellow"></a>

In this repository, I implemented semantic segmentation inference code by using SegFormer model from hugging face.
  
</div>

<p align="center">
  <img src="https://user-images.githubusercontent.com/71377772/216529315-3fbcddf6-4c05-431e-8744-a0055029aa20.png" width=75% height=75% />
</p>

# model 
You can choose a pre-trained segformer model from the following list, and designate when running `segmentation.py --model model_name`
```
[
  'nvidia/segformer-b5-finetuned-cityscapes-1024-1024',  
  'nvidia/segformer-b5-finetuned-ade-640-640',
  'nvidia/segformer-b4-finetuned-cityscapes-1024-1024',  
  'nvidia/segformer-b4-finetuned-ade-512-512',
  'nvidia/segformer-b3-finetuned-cityscapes-1024-1024',  
  'nvidia/segformer-b3-finetuned-ade-512-512',
  'nvidia/segformer-b2-finetuned-cityscapes-1024-1024',  
  'nvidia/segformer-b2-finetuned-ade-512-512',
  'nvidia/segformer-b1-finetuned-cityscapes-1024-1024',  
  'nvidia/segformer-b1-finetuned-ade-512-512',
  'nvidia/segformer-b0-finetuned-cityscapes-1024-1024',
  'nvidia/segformer-b0-finetuned-cityscapes-512-1024',  
  'nvidia/segformer-b0-finetuned-cityscapes-640-1280',  
  'nvidia/segformer-b0-finetuned-cityscapes-768-768',  
  'nvidia/segformer-b0-finetuned-ade-512-512'
]
```
