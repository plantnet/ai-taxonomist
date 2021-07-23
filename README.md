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
