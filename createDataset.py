import gbif_dl
import argparse
import json
import os

import torch
from torchvision import transforms, datasets

def main():
    parser = argparse.ArgumentParser(description='Cos4Cloud dataset creation')
    parser.add_argument('-s', '--species', type=argparse.FileType('r'),
                        help='json file listing species ids')
    parser.add_argument('-p', '--providers', type=argparse.FileType('r'),
                        help='json file listing providers ids')
    parser.add_argument('-d', '--data', help='where to store images')
    parser.add_argument('-n', '--number', type=int, metavar='N', default=1000,
                        help='Number of images per species')

    args = parser.parse_args()

    batch_size=16
    # with open(args.species, 'r') as speciesFile:
    #     with open(args.providers, 'r') as providersFile:
    os.makedirs(args.data, exist_ok=True)
    species = json.load(args.species)
    providers = json.load(args.providers)
    queries = {'speciesKey': species, 'datasetKey': providers}

    urls = []
    if False:
        data_generator = gbif_dl.api.generate_urls(queries=queries, label='speciesKey', nb_samples_per_stream=args.number,
                                                   split_streams_by=['speciesKey'], mediatype='StillImage')
        for i in data_generator:
            print(i)
            urls.append(i)
        with open(args.data+"/urls.json", 'w') as f:
            json.dump(urls, f)
    else:
        with open(args.data + "/urls.json", 'r') as f:
            urls = json.load(f)

    img_dir = os.path.join(args.data, 'img')
    gbif_dl.io.download(urls, root=img_dir, random_subsets={'train':0.95, 'val':0.05},
                        nb_workers=4, retries=1, verbose=True)

    if True:
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(303),
                transforms.RandomCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(303),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        image_datasets = {x: datasets.ImageFolder(os.path.join(img_dir, x), data_transforms[x])
                          for x in ['train', 'val']}
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                      shuffle=True, num_workers=8)
                       for x in ['train', 'val']}
        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
        # class_names = image_datasets['train'].classes
        print(dataset_sizes)


if __name__=="__main__":
    main()