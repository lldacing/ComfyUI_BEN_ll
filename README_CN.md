[English](README.md)

BEN 是一种深度学习模型，旨在自动从图像中删除背景，从而生成蒙版和前景图像。

## 预览
![save api extended](doc/base.png)

## 安装

- 手动安装
```shell
    cd custom_nodes
    git clone https://github.com/lldacing/ComfyUI_Ben_ll.git
    cd ComfyUI_Ben_ll
    pip install -r requirements.txt
    # 重启comfyUI
```
    

## 模型
从[HuggingFace](https://huggingface.co/PramaLLC/BEN/resolve/main/BEN_Base.pth?download=true)下载模型放到目录`ComfyUI/models/rembg/ben/`


## 感谢

原项目 [BEN](https://huggingface.co/PramaLLC/BEN)

