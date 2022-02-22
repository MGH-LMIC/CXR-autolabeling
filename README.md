# Auto-labeling Method on CXR

Automated Labeling Based on Similarity to a Validated CXR AI Model Using a Quantitative, Explainable, “Smart” Atlas-Based Approach.

## Prerequisites

- Python 3.7.+
- PyTorch 1.2.0+
- Python Packages: ./environment.yml


## Test

- You can use 'autolabeling.py' to test our auto-labeling method to 490 external public datasets (CheXpert[1], MIMIC[2], and NIH[3])
- You have to set up gpus

```
$ python ./autolabeling.py --cuda=<comma-separated gpu-ids>
```


## Data & Result sharing
- We share the data and results of the autolabeling method to https://github.com/MGH-LMIC/AutoLabels-PublicData-CXR-PA.git:

         * Platinum labels for 490 external public datasets (platinum_label_<abnormal feature>.csv)
        
         * Automatic labels applied to posteroanterior (PA) images in CheXpert (n=224,316), MIMIC (n=377,110), and NIH (n=108,948)


## Demo (w/ Docker)

### 1. Docker and Nvidia-docker installation
In order to load our docker image, you need to install `Docker` and `Nvidia-docker`.
- Docker : 18.03.0-ce ([installation](https://docs.docker.com/install/linux/docker-ce/ubuntu/#os-requirements))
- Nvidia-docker : v2.0.3 ([installation](https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(version-2.0))))

Please follow the instructions described in the above links. Estimated installation time is about **10** minutes.

### 2. Build image
You have to follow up the below instruction to make docker image

- Clone this repo
- Download the pre-trained models and atlas for demo from Dropbox

```sh
# Dowload autolabeling zip file
$ wget -O autolabeling.zip https://www.dropbox.com/s/0iur74nd987qube/autolabeling.zip?dl=0
# unzip the zip file
$ unzip autolabeling.zip

# Dowload data zip file
# request to authors to get a permission
# unzip the zip file
$ unzip data.zip

# Remove files
$ rm autolabeling.zip data.zip
```
- Make docker
```sh
$ sudo nvidia-docker build . -t lmic-autolabeling:1.0 -f Dockerfile
```

### 3. Run image
Open the terminal
```sh
$ export WORK_DIR="<absolute path of ./>"
$ export OUTPUT_DIR="$WORK_DIR/autolabeling"
# Run docker image
$ sudo nvidia-docker run --gpus all -v $OUTPUT_DIR:/home/docker/autolabeling --shm-size 8G --name autolabel -it lmic-autolabeling:1.0 /bin/bash

```

### 4. Run demo
Now, you are ready to enjoy our Auto-labeling Method on CXRs in a conda environment of the container!

```sh
$ python autolabeling.py --cuda=<comma-separated gpu-ids>
```

**[Output]**
Automated labels are returned as output_autolabeling.csv in autolabeling folder after the program is executed.
```
columns ={
         '{class}_gt'  : platinum ground truth for public open datasets,
         '{class}_pr'  : predicted probability by xAI model,
         '{class}_ps'  : patch similarity,
         '{class}_cf'  : confidence,
         '{class}_pSim': probability of similarity (key metric for auto-labeling method),
         '{class}_agt' : auto-labeling results using the optimal pSim thresholds
         }
```


## Reference
[1] Irvin, J., Rajpurkar, P., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K. and Seekins, J. Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. In Proceedings of the AAAI Conference on Artificial Intelligence 33, 590-597 (2019). Available at: https://stanfordmlgroup.github.io/competitions/chexpert/

[2] Johnson, A.E., Pollard, T.J., Shen, L., Li-Wei, H.L., Feng, M., Ghassemi, M., Moody, B., Szolovits, P., Celi, L.A. and Mark, R.G. MIMIC-III, a freely accessible critical care database. Scientific data, 3(1), 1-9 (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635

[3] Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. M. Chestx-ray8: Hospital-scale chest x-ray database and benchmarks on weakly-supervised classification and localization of common thorax diseases. In Proceedings of the IEEE conference on computer vision and pattern recognition, 2097-2106 (2017). Images are available via Box: https://nihcc.app.box.com/v/ChestXray-NIHCC
