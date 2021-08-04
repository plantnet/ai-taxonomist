# c4c-identify

Automatically generate a Pl@ntNet like identification engine from a list of species, ou presque

## Requirements:
    * pytorch
    * gbif-dl (pip install gbif-dl)


`--help` is your friend !

## Example
   The `frogs` directory contains some gbif species id for the genus `Rana` (ie frogs) and associated data providers

```bash
python createDataset.py --species frogs/rana.json --providers frogs/providers.json --directory /data/frogs
python train.py /data/frogs --arch mobilenet_v2
```


# How to build an engine

## Crawl GBIF

### Retrieve species id from canonical names (optional)
```
python canoncicalName2speciesId.py --data /path/to/data --names /path/to/names.json
```
will create a `species_id.json` in `/path/to/data/` with the coresponding gbif species id

### Retrieve images urls
```
python createDataset.py --data /path/to/data --species species_id.json --providers providers.json --number N
```


# How to build an engine
produce a data set with at max N image per species

## Train a deep neural network
```
python train.py /data/frogs --arch mobilenet_v2

```


## Build a Snoop and its groundtruth
### Create groundtruth

```
python createGT.py --data /path/to/data
```

### Create configuration files

```
python createConfig.py --data /path/to/data
```

### build the snoop
```
cd /path/to/data
snoopCorpus --root . --corpus c4c --recurse --input img
snoopExtractor --root . --corpus c4c --feature Feature --doc_plugin Config/doc_ocv.json --desc_plugin Config/desc_torchscript.json --min_num_desc 1 --nb_lot_per_thread 100 --nb_threads 10
snoopDatabase --root . --corpus c4c --feature Feature --database Index --db_plugin Config/db_pmh.json
```
