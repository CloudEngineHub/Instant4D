# Instant4D: 4D Gaussian Splatting in Minutes



## Table of Content
---

## Clone

Make sure to clone the repository with the submodules by using:
`git clone --recursive git@github.com:Zhanpeng1202/Instant4D-Dev.git`

### 4DGS Remote Viewer
---
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

### Get Started 
---

### Environment

The hardware and software requirements are the same as those of the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), which this code is built upon. To setup the environment, please run the following command:

```shell
git clone https://github.com/fudan-zvg/4d-gaussian-splatting
cd 4d-gaussian-splatting
conda env create --file environment.yml
conda activate 4dgs
```


### Reproduce
---
We provide the GPU 












### Acknowledgement
---

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!

- [Gaussian-Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
- [4d-gaussian-splatting](https://github.com/fudan-zvg/4d-gaussian-splatting)
- [Mega-SAM](https://github.com/mega-sam/mega-sam)
