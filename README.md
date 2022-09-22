# ai-taxonomist

(almost) Automatically generate a Pl@ntNet like identification engine from a GBIF occurrences Darwin Core Archive

By using this tool, the user agree to the GBIF data user agreement (https://www.gbif.org/terms/data-user)


## Required environment
* python 3.9
* docker >=19.03.12
* Cuda capable GPUs
  * cuda >=11.0,  
  * nvidia-docker >=1.0.1 (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)
* github.com access
* GitHub docker register access

 > `--help` is your friend !  
Every command listed bellow answer the --help flag, use it!
## Offline preparation
### Python phase
#### Requirements:
* pytorch (https://pytorch.org/get-started/locally/ - torchaudio is not needed)
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
```bash
python -u train.py --data /path/to/dataset --arch resnet18 --lr 0.01 --batch-size 256 \
  --epoch 50 --patience 4 --print 50 --workers 4 --pretrained --imbalanced \
  --dist-url tcp://127.0.0.1:9123 --dist-backend nccl --multiprocessing-distributed --world-size 1 --rank 0 \
  |& tee torch_resnet18.log
python scriptModel.py --data /path/to/dataset --arch resnet18
```
**Warning!** 
* It will also probably take a while
* Do not forget to script the trained model with `scriptModel.py` once the training is completed

Optionally, you might plot the training progress (and adjust your parameters) with `progressPlot.py` (requires regex - pip install regex):
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
As executable are called through docker, it is easier do define `DOCKER_COMMAND` as follows:
  * CPU: 
```bash
export DOCKER_IMAGE="ghcr.io/plantnet/ai-taxonomist/cpu:latest"
export DOCKER_COMMAND="docker run -it --rm -v /path/to/data:/data --name ait-builder $DOCKER_IMAGE" 
```
  * GPU:
```bash
export DOCKER_IMAGE="ghcr.io/plantnet/ai-taxonomist/gpu:latest"
export DOCKER_COMMAND="docker run -it --rm --gpus all -v /path/to/data:/data --name ait-builder $DOCKER_IMAGE" 
```

**Note:** According to your configuration (ie you are not in the `docker` group), you might need to `sudo` the `docker` command

### Build the identification engine
```bash
$DOCKER_COMMAND /opt/snoop/bin/snoopCorpus --root /data/ai-taxonomist --corpus c4c --recurse --input /data/img
$DOCKER_COMMAND /opt/snoop/bin/snoopExtractor --root /data/ai-taxonomist --corpus c4c --feature Feature \
    --doc_plugin Config/doc_ocv.json --desc_plugin Config/desc_torchscript.json \
    --min_num_desc 1 --nb_lot_per_thread 100 --nb_threads 10
$DOCKER_COMMAND /opt/snoop/bin/snoopDatabase --root /data/ai-taxonomist --corpus c4c --feature Feature \
    --database Index --db_plugin Config/db_pmh.json
```

**Notes:**
* When you are using a small GPU and a large model `snoopExtractor` might fail by lack of GPU memory. In this case, you diminish the `--nb_threads` parameter (and may be `--nb_lot_per_thread`). 
Then rerun `snoopExtractor` with the added `--rescue` flag, and once it succeed rerun `snoopDatabase`
* if your dataset does not have enough images (ie. less than 2k), `snoopDatabase` will fail. In this case the identification results illustration by most similar images won't be available, but you still can run image classification by adding the `--no-illustrate` flag to the `c4cIdentify` command bellow.

## Online identification

While using one or more GPU is mandatory for the preparation phase, it is optional for the online identification service.
`ai-taxonomist` will run on CPU with the same accuracy (but with longer response time) as on GPU.

To run the `ai-taxonomist` REST API on port `PORT` on a production computer, you need to `rsync` (or any tool which preserve symbolic links) the `/path/to/data/ai-taxonomist` directory produced by previous steps on the production computer.
### Requirements:
* docker >=19.03.12
* optionally nvidia-docker >=1.0.1 to run on GPU (recommended with large dataset)
* the latest ai-taxonomist image
* the ai-taxonomist directory (and its content!) produced by the offline preparation


### Setup
As executable are called through docker, it is easier do define `DOCKER_RUN` as follows:
  * CPU: 
```bash
export DOCKER_IMAGE="ghcr.io/plantnet/ai-taxonomist/cpu:latest"
export DOCKER_RUN="docker run -it --rm -p PORT:8080 -v /path/to/ai-taxonomist:/data --name ai-taxonomist $DOCKER_IMAGE" 
```
  * GPU:
```bash
export DOCKER_IMAGE="ghcr.io/plantnet/ai-taxonomist/gpu:latest"
export DOCKER_RUN="docker run -it --rm --gpus all -p PORT:8080 -v /path/to/ai-taxonomist:/data --name ai-taxonomist $DOCKER_IMAGE" 
```

### Launch the engine
```bash
$DOCKER_RUN /opt/snoop/bin/c4cIdentify --data /data
```

The REST server is now accessible through `http://0.0.0.0:PORT`.  
Main routes are
* (GET)  `/help` list available routes and their usage
* (GET)	 `/identify?image=image_uri[&image=image_uri...]` identify the given image(s)
* (POST) `/identify` identify the image(s) sent as 'image:' in the message body

### User Interface
The `ai-taxonomist` REST service is compatible with the `ai-taxonomist-webcomponent` (https://github.com/plantnet/ai-taxonomist-webcomponent)

## Example
To illustrate the usage of `ai-taxonomist`, we built a DWCA from an occurrence search on GBIF on the order `Anura`. Following the above steps with `--doi 10.15468/dl.epcnam` produced a dataset with 3k frog species illustrated by 500k images. 

> GBIF.org (07 October 2021) GBIF Occurrence Download  https://doi.org/10.15468/dl.epcnam

This example is running on https://c4c.inria.fr/demo/ as a demonstration of `ai-taxonomist` and its web component 

The REST api is accessible at https://c4c.inria.fr/api/
