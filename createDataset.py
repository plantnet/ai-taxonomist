import gbif_dl
import argparse
import json
import os

import torch
from torchvision import transforms, datasets

def main():
    parser = argparse.ArgumentParser(description='Cos4Cloud dataset creation')
    parser.add_argument('-s', '--species', type=argparse.FileType('r'),
                        help='json file listing species ids', required=True)
    parser.add_argument('-p', '--providers', type=argparse.FileType('r'),
                        help='json file listing providers ids')
    parser.add_argument('-d', '--data', help='where to store images', required=True)
    parser.add_argument('-n', '--number', type=int, metavar='N', default=1000,
                        help='Number of images per species')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--no-crawl', action='store_false', dest='crawl',
                        help='Crawl GBif to get urls, or use the store data/url.json file')
    parser.add_argument('--no-download', action='store_false', dest='download',
                        help='Download images or only store the data/url.json file and exit')

    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)

    urls = []
    if args.crawl:
        species = json.load(args.species)
        providers = json.load(args.providers)
        queries = {'speciesKey': species, 'datasetKey': providers}
        data_generator = gbif_dl.api.generate_urls(queries=queries, label='speciesKey', nb_samples_per_stream=args.number,
                                                   split_streams_by=['speciesKey'], mediatype='StillImage')
        print('Retrieving urls...')
        for i in data_generator:
            if args.verbose:
                print(i)
            urls.append(i)
            if len(urls) % 500 == 0:
                print(len(urls),'url retrieved')
        with open(args.data+"/urls.json", 'w') as f:
            json.dump(urls, f)
        print(len(urls), 'urls saved in', args.data + "/urls.json")
    else:
        with open(args.data + "/urls.json", 'r') as f:
            urls = json.load(f)
        print(len(urls), 'urls loaded in', args.data+"/urls.json")

    if args.download:
        img_dir = os.path.join(args.data, 'img')
        print("Downloading images to", img_dir)
        gbif_dl.io.download(urls, root=img_dir, random_subsets={'train':0.95, 'val':0.05},
                            nb_workers=1, retries=1, verbose=True)
        print('Images downloaded')
    else:
        print('Download skipped')


if __name__=="__main__":
    main()