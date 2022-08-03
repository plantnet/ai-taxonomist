# ai-taxonomist

(almost) Automatically generate a Pl@ntNet like identification engine from a GBIF occurrences Darwin Core Archive

## Required environment
* python 3.9
* docker >=19.03.12
* Cuda capable GPUs
  * cuda >=11.0,  
  * nvidia-docker >=1.0.1 (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
* github.com access
* github docker register access

 > `--help` is your friend !  
Every command listed bellow answer the --help flag, use it!
## Offline preparation
### Python phase
#### Requirements:
* pytorch (https://pytorch.org/get-started/locally/)
* gbif-dl (pip install gbif-dl)
* scipy (pip install scipy)
* img2dataset (pip install img2dataset)

#### Dataset creation

```commandline
python createDataset.py --data /path/to/dataset --doi prefix/suffix
```
Create a dataset in /path/to/dataset (must be writeable) base on the given prefix/suffix GBIF occurrences Darwin Core Archive doi

**Warning!** It will probably take a while, use several GB of disk and network to
  * retrieve the Darwin Core Archive and eventually crawl GBIF on the targeted species 
  * download and store in /path/to/dataset/img all the images referenced by the doi
  * split the downloaded images into a train and a validation set


Useful options:
* `--no-crawl`: use the stored dwca instead of downloading it
* `--no-download`: use the stored images instead of downloading them
* `--no-split`: keep all the downloaded images in place instead of creating a train/val split

To see all available options: ```python createDataset.py --help```

#### Neural network training
On a multi gpu computer:
```commandline
python -u train.py --data /path/to/dataset --arch resnet18 --lr 0.01 --batch-size 256 \
  --epoch 50 --patience 4 --print 50 --workers 4 --pretrained --imbalanced \
  --dist-url tcp://127.0.0.1:9123 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 \
  |& tee torch_resnet18.log
python scriptModel.py --data /path/to/dataset --arch resnet18
```
**Warning!** 
* It will also probably take a while
* Do not forget to script the trained model with `scriptModel.py`

Optionally, you might plot the training progress (and adjust your parameters) with `progressPlot.py`:
```commandline
python progressPlot.py --train torch_resnet18.log
```

#### Build Snoop's configuration files
```commandline
python createConfig.py --data /path/to/dataset
python createGT.py --data /path/to/dataset
```

### Docker phase
#### Requirements:
* docker >=19.03.12
* optionally nvidia-docker >=1.0.1 to run on GPU (recommended with large dataset)
* the latest ai-taxonomist image

####  Setup
As executable are called through docker, it is easier do define the `DOCKER_COMMAND` as follows:
  * CPU: 
```bash
export DOCKER_IMAGE="ghcr.io/plantnet/ai-taxonomist/cpu:latest"
export DOCKER_COMMAND="docker run -it --rm -v /path/to/data:/data --name ait-builder $DOCKER_IMAGE" 
```
  * GPU:
```bash
export DOCKER_IMAGE="ghcr.io/plantnet/ai-taxonomist/cu113:latest"
export DOCKER_COMMAND="docker run -it --rm --gpus all -v /path/to/data:/data --name ait-builder $DOCKER_IMAGE" 
```

### Build the identification engine
```bash
$DOCKER_COMMAND /opt/snoop/bin/snoopCorpus --root /data/ai-taxonomist --corpus c4c --recurse --input /data/img
$DOCKER_COMMAND /opt/snoop/bin/snoopExtractor --root /data/ai-taxonomist --corpus c4c --feature Feature \
    --doc_plugin Config/doc_ocv.json --desc_plugin Config/desc_torchscript.json \
    --min_num_desc 1 --nb_lot_per_thread 100 --nb_threads 10
$DOCKER_COMMAND /opt/snoop/bin/snoopDatabase --root /data/ai-taxonomist --corpus c4c --feature Feature \
    --database Index --db_plugin Config/db_pmh.json
```
-----------
TO BE CONTINUED

------------

## Examples

```bash
python createDataset.py --data /path/to/dataset --doi 10.15468/dl.epcnam
```
Create a dataset in /path/to/dataset (must be writeable) base on the given GBIF occurrences Darwin Core Archive doi

**Warning!** It will take a while, use several GB of disk and network to
  * crawl GBIF on the `Anura` order and retrieve information about ~ 3k frog species 
  * download more than half a million images to your hard drive
  * split the downloaded images into a train and a validation set




The `frogs` directory contains required data to build a c3c-identification engine on the 'Rana' genus (frogs).
It is declined accordingly to how familiar you are with `gbif`
* novice: no prior knowledge of gbif required
* intermediate: require to visit https://www.gbif.org/ and pick some gbif species ids and related datasets ids
* expert: for those able to make complex gbif request and save them as DOI 

## Offline preparation

### Create the C4C Dataset `createDataset.py`
This stage will extract needed information from `gbif`, download occurrences images and create a training and a validation set in `/data/frogs/imgs`

#### `frogs/novice`
The file `species_names.txt` contains a list of scientific names of species from the `Rana` genus, 1 name per line

```bash
python createDataset.py --data /data/frogs --names frogs/novice/species_names.txt --number 500
```

#### `frogs/intermediate`

Json files `species_id.json` and `providers.json` respectively list some `Rana` species ids and some dataset ids

```bash
python createDataset.py --data /data/frogs --species frogs/intermediate/species_id.json --providers frogs/intermediate/providers.json --number 500
```

#### `frogs/expert`
A gbif expert has prepared a query: https://doi.org/10.15468/dl.bx6xeq

```bash
python createDataset.py --data /data/frogs --doi 10.15468/dl.bx6xeq
```

### Train and export the DNN model

/!\ Using a computer with several cuda gpus is highly recommended 

For now, available DNN architectures are
* densenet121, densenet161, densenet169, densenet201
* inception_v3
* mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
* resnet101, resnet152, resnet18, resnet34, resnet50, resnext101_32x8d, resnext50_32x4d, wide_resnet101_2, wide_resnet50_2

```bash
python train.py --data /data/frogs --arch mobilenet_v2
python scriptModel.py --data /data/frogs
```

### Build a Snoop and its groundtruth

* CPU Only:
```bash
export DOCKER_COMMAND="docker run -it --rm -v /data/frogs:/data registry.gitlab.inria.fr/snoop/c4c-identify/cpu:latest"
```
* GPU (cuda 11.1)
```bash
export DOCKER_COMMAND="docker run -it --rm --gpus all -v /data/frogs:/data registry.gitlab.inria.fr/snoop/c4c-identify/cu111:latest"
```


1. generate configuration files
```bash
python createConfig.py --data /data/frogs
```

2. create the _Snoop_ corpus
```bash
$DOCKER_COMMAND /opt/snoop/bin/snoopCorpus --root /data --corpus c4c --recurse --input img
```

3. extract descriptors (gpu recommended)
```bash
$DOCKER_COMMAND /opt/snoop/bin/snoopExtractor --root /data --corpus c4c --feature Feature \
    --doc_plugin Config/doc_ocv.json --desc_plugin Config/desc_torchscript.json --min_num_desc 1 --nb_lot_per_thread 100 --nb_threads 10
```
4. compute the index

```bash
$DOCKER_COMMAND /opt/snoop/bin/snoopDatabase --root /data --corpus c4c --feature Feature --database Index --db_plugin Config/db_pmh.json
```

## Online: launch the identification service

* CPU Only:
```bash
export DOCKER_COMMAND="docker run -it --rm -v /data/frogs:/data registry.gitlab.inria.fr/snoop/c4c-identify/cpu:latest"
```
* GPU (cuda 11.1)
```bash
export DOCKER_COMMAND="docker run -it --rm --gpus all -v /data/frogs:/data registry.gitlab.inria.fr/snoop/c4c-identify/cu111:latest"
```

```bash
$DOCKER_COMMAND /opt/snoop/bin/c4cIdentify --data /data
```
