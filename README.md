# Instant4D: 4D Gaussian Splatting in Minutes

<a href="https://instant4d.github.io/"><img src='https://img.shields.io/badge/Website-Instant4D-green' alt='Website'></a>
<a href="#citation"><img src='https://img.shields.io/badge/BibTex-Instant4D-blue' alt='Paper BibTex'></a>



## Installation

Make sure to clone the repository with the submodules by using:
`git clone --recursive git@github.com:Zhanpeng1202/Instant4D-Dev.git`

### Environment

The hardware and software requirements are the same as those of the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
conda env create --file environment.yml
conda activate instant4D

cd SLAM/mega-samn/base
python setup.py install
cd ../../../../
```




### 4DGS Remote Viewer

We provide a lightweight websocket remote viewer to visualize 4DGS training process. Users can train 4DGS on a server and hope to view it on local computer.

On the local computer

```shell
# download these file in local computer
git clone git@github.com:Zhanpeng1202/gaussian_splatting_websocket_viewer.git

# Connect Server with SSH with vscode
vscode ssh server 

#set up forward port in vscode
Terminal -> Ports -> Forward a Ports -> 6119
```

On the server

```shell
# clone the official gaussain splatting repository
git clone git@github.com:graphdeco-inria/gaussian-splatting.git --recursive

# put networkGUI_Websocket.py in to correct location inside the cloned repository
<location>
|---gaussian_splatting
|   |---gaussain_render
|   |   |---network_gui.py
|   |   |---network_gui_websocket.py

|   |---train.py 
# replace train.py with that provided in this repository
```





### Dataset

```
mkdir dataset
cd dataset
```

### [Nvidia](https://gorokee.github.io/jsyoon/dynamic_synth/)

Download the pre-processed data by [DynamicNeRF](https://github.com/gaochen315/DynamicNeRF).
```
mkdir Nvidia
wget --no-check-certificate https://filebox.ece.vt.edu/~chengao/free-view-video/data.zip
unzip data.zip
rm data.zip
```

### [DAVIS](https://davischallenge.org/davis2016/code.html) or custom sequences

Put the images in the following data structure.
```
RoDynRF                
├── dataset
│   ├── DAVIS_480p
│   │   ├── bear
│   │   │   └── images
│   │   │       ├── 00000.jpg
│   │   │       ├── 00001.jpg
│   │   │       ├── ...
│   │   │       └── 00081.jpg
│   │   ├── blackswan
│   │   ├── ...
│   │   └── train
│   ├── custom
│   │   ├── sequence_000
│   │   │   └── images
│   │   │       ├── 00000.jpg
│   │   │       ├── 00001.jpg
│   │   │       ├── ...
```
Run the following preprocessing steps.


## Reproduction

We provide the input file after grid prunning to facilitate reproduce, the processed data can be find here 
[data](https://drive.google.com/drive/u/1/folders/1Ce4C0WpabtTQvZXeiUmMKVhD5kA9wJPf)


### Optimization

Change the `source_path` and `model_path` accordingly in the config files, then run 
```
python batch_train.py
```


## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
- [Mega-SAM](https://github.com/mega-sam/mega-sam)

## Citation

If you find this project useful in your research, please consider citing:
```
@article{luo2025instant4d,
  title={Instant4d: 4d gaussian splatting in minutes},
  author={Luo, Zhanpeng and Ran, Haoxi and Lu, Li},
  journal={Advances in neural information processing systems},
  year={2025}
}
```
