## U-GAT-IT &mdash; Unofficial MXNet Implementation
This project is an unofficial MXNet implementation of U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation. It is adapted from  [znxlwm/UGATIT-pytorch](https://github.com/znxlwm/UGATIT-pytorch).

### This MXNet implementation works in process. There are still some problems, so I do not recommend you to train it. I will update the README after fixing the bugs. I hope training the LARGE model under 11Gb GPU memory by MXNet's black technology : )

### : Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation

<div align="center">
  <img src="./assets/teaser.png">
</div>

### [Paper](https://arxiv.org/abs/1907.10830) | [Official Tensorflow code](https://github.com/taki0112/UGATIT)
The results of the paper came from the **Tensorflow code**


> **U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation**<br>
> **Junho Kim (NCSOFT)**, Minjae Kim (NCSOFT), Hyeonwoo Kang (NCSOFT), Kwanghee Lee (Boeing Korea)
>
> **Abstract** *We propose a novel method for unsupervised image-to-image translation, which incorporates a new attention module and a new learnable normalization function in an end-to-end manner. The attention module guides our model to focus on more important regions distinguishing between source and target domains based on the attention map obtained by the auxiliary classifier. Unlike previous attention-based methods which cannot handle the geometric changes between domains, our model can translate both images requiring holistic changes and images requiring large shape changes. Moreover, our new AdaLIN (Adaptive Layer-Instance Normalization) function helps our attention-guided model to flexibly control the amount of change in shape and texture by learned parameters depending on datasets. Experimental results show the superiority of the proposed method compared to the existing state-of-the-art models with a fixed network architecture and hyper-parameters.*

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── trainA
           ├── xxx.jpg (name, format doesn't matter)
           ├── yyy.png
           └── ...
       ├── trainB
           ├── zzz.jpg
           ├── www.png
           └── ...
       ├── testA
           ├── aaa.jpg 
           ├── bbb.png
           └── ...
       └── testB
           ├── ccc.jpg 
           ├── ddd.png
           └── ...
```

### Train
```
> python main.py --dataset selfie2anime
```
* If the memory of gpu is **not sufficient**, set `--light` to 1

### Test
```
> python main.py --dataset selfie2anime --phase test
```

## Architecture
<div align="center">
  <img src = './assets/generator.png' width = '785px' height = '500px'>
</div>

---

<div align="center">
  <img src = './assets/discriminator.png' width = '785px' height = '450px'>
</div>

## Results
### Ablation study
<div align="center">
  <img src = './assets/ablation.png' width = '438px' height = '346px'>
</div>

### User study
<div align="center">
  <img src = './assets/user_study.png' width = '738px' height = '187px'>
</div>

### Comparison
<div align="center">
  <img src = './assets/kid.png' width = '787px' height = '344px'>
</div>

## Citation
If you find this code useful for your research, please cite the paper:

```
@misc{kim2019ugatit,
    title={U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation},
    author={Junho Kim and Minjae Kim and Hyeonwoo Kang and Kwanghee Lee},
    year={2019},
    eprint={1907.10830},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

## Author
[Junho Kim](http://bit.ly/jhkim_ai), Minjae Kim, Hyeonwoo Kang, Kwanghee Lee
