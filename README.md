# c4c-identify

Automatically (ou presque) generate a Pl@ntNet like identification engine from a list of species

-------------
This is probably outdated, see https://gitlab.inria.fr/snoop/c4c-identify/-/wikis/home
------------

## Requirements:
* pytorch (https://pytorch.org/get-started/locally/)
* gbif-dl (pip install gbif-dl)
* scipy (pip install scipy)
* a working docker (>19.03) and a _Snoop_ docker image
    * registry.gitlab.inria.fr/snoop/c4c-identify/cpu:latest to run on cpu only
    * registry.gitlab.inria.fr/snoop/c4c-identify/cu111:latest to run on cuda (at least 11.1) gpus 
        * nvidia-docker is also required in this case


 > `--help` is your friend !

## Examples
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
