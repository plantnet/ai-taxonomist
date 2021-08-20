import sys
import gbif_dl
import argparse
import json
import os
import pygbif.species.name_backbone as gbif_species

# Used by the single thread download
from gbif_dl.io import MediaData
import requests
from typing import AsyncGenerator, Callable, Generator, Union, Optional
from pathlib import Path
import random
import hashlib
import filetype

def crawlNames(args: dict) -> list:
    species_ids = []
    other = []
    cnt = 0
    for n in args.names:
        n = n.rstrip()
        data = gbif_species(name=n, rank='species', verbose=args.verbose)
        # status = data.get('status', 'unknown')
        match = data.get('matchType', 'unknown')
        rank = data.get('rank', None)
        id = data.get('speciesKey', None)

        if match == 'EXACT' and rank == 'SPECIES' and id:
            species_ids.append(id)
        else:
            print(n, 'does not match any species in gbif', {'name': n, 'rank': rank, 'match': match})
            if args.verbose:
                print(data)
            other.append({'name': n, 'gbif': data})
        cnt += 1
        if args.verbose and cnt % 50 == 0:
            print(cnt, 'species matched')
    else:
        print('Unable to get gbif info for name', n)

    if len(other):
        output = os.path.join(args.data,'no_species.json')
        print('some species names do not match any species, see', output, 'for details')
        with open(output, 'w') as f:
            json.dump(other, f)

    with open(os.path.join(args.data, 'species_ids.json'), 'w') as f:
        json.dump(species_ids, f)

    return species_ids


def downloadMedia(item: MediaData, root: str, random_subsets: dict ={'train': 0.95, 'val': 0.05},
                  is_valid_file: Optional[Callable[[bytes], bool]] = None,
                  overwrite: bool = False,
                  proxy: Optional[str] = None,
                  ) -> None:
    if isinstance(item, dict):
        url = item.get("url")
        basename = item.get("basename")
        label = item.get("label")
        subset = item.get("subset")
    else:
        url = item
        label, basename, subset = None, None, None

    if subset is None and random_subsets is not None:
        subset_choices = list(random_subsets.keys())
        p = list(random_subsets.values())
        subset = random.choices(subset_choices, weights=p, k=1)[0]

    label_path = Path(root)

    if subset is not None:
        label_path /= Path(subset)

    # create subfolder when label is a single str
    if isinstance(label, str):
        # append label path
        label_path /= Path(label)

    label_path.mkdir(parents=True, exist_ok=True)

    if basename is None:
        # hash the url
        basename = hashlib.sha1(url.encode("utf-8")).hexdigest()

    check_files_with_same_basename = label_path.glob(basename + "*")
    if list(check_files_with_same_basename) and not overwrite:
        # do not overwrite, skips based on base path
        return

    proxies = None
    if proxy:
        proxies={'http':proxy, 'https':proxy}
    res = None
    try:
        res = requests.get(url, proxies=proxies)
    except Exception as e:
        print('Failed to download', url,':', e)
        return

    # Check everything went well
    if res.status_code != 200:
        print(f"Download failed: {res.status_code}")
        return

    content = res.content

    # guess mimetype and suffix from content
    kind = None
    try:
        kind = filetype.guess(content)
    except Exception as e:
        print('Failed to find type of', url, ':', e)
        return

    if kind is None:
        return
    else:
        suffix = "." + kind.extension
        mime = kind.mime


    if is_valid_file is not None:
        if not is_valid_file(content):
            print(f"File check failed")
            return

    file_base_path = label_path / basename
    file_path = file_base_path.with_suffix(suffix)
    with open(file_path, 'wb') as f:
        f.write(content)
    if isinstance(label, dict):
        json_path = (label_path / item["basename"]).with_suffix(".json")
        with open(json_path, mode="+w") as fp:
            fp.write(json.dumps(label))



def main():
    parser = argparse.ArgumentParser(description='Cos4Cloud dataset creation')
    required = parser.add_argument_group('required argument')
    required.add_argument('--data', help='where to store dataset information and images', required=True)
    parser.add_argument('-n', '--number', type=int, metavar='N', default=1000,
                        help='Number of images per species')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('--no-crawl', action='store_false', dest='crawl',
                        help='do not crawl gbif, use the stored DATA/images.json file instead')
    parser.add_argument('--no-download', action='store_false', dest='download',
                        help='do not download images')
    parser.add_argument('--single-thread-dl', action='store_true', default=False)
    novice = parser.add_argument_group(title='novice usage', description='requires no previous knowledge of gbif')
    novice.add_argument('--names', type=argparse.FileType('r'),
                        help='text file with one species canonical name per line')
    intermediate = parser.add_argument_group(title='intermediate usage', description='requires some knowledge of gbif')
    intermediate.add_argument('--species', type=argparse.FileType('r'),
                        help='json file listing gbif species ids')
    intermediate.add_argument('--providers', type=argparse.FileType('r'),
                        help='json file listing gbif providers ids')
    expert = parser.add_argument_group(title='expert', description='requires to fully understand gbif')
    expert.add_argument('--doi', type=str, help="a gbif's query doi")

    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)

    species_ids = []
    if args.crawl:
        if not args.doi:
            if args.names:
                if args.species:
                    print('it does not make sens to provide both --name and --species arguments')
                    sys.exit(-1)
                species_ids = crawlNames(args)
            elif args.species:
                species_ids = json.load(args.species)
            #handle doi

            if args.providers:
                providers = json.load(args.providers)
                queries = {'speciesKey': species_ids, 'datasetKey': providers}
            else:
                queries = {'speciesKey': species_ids}
            data_generator = gbif_dl.api.generate_urls(queries=queries, label='speciesKey',
                                                       nb_samples_per_stream=args.number,
                                                       split_streams_by=['speciesKey'],
                                                       mediatype='StillImage',
                                                       one_media_per_occurrence=False,
                                                       verbose=args.verbose)
        else: # use a precomputed gbif query from its doi
            if args.names or args.species:
                print('it does not make sens to provide both --doi and --name or --species arguments')
                sys.exit(-1)
            data_generator = gbif_dl.dwca.generate_urls(args.doi,
                                                        dwca_root_path=os.path.join(args.data, "dwcas"),
                                                        label="speciesKey",
                                                        mediatype='StillImage',
                                                        one_media_per_occurrence=False,)

        print('Retrieving image metadata...')
        urls = []
        for i in data_generator:
            urls.append(i)
            if len(urls) % 500 == 0:
                print(len(urls),'image metadata retrieved')
        with open(args.data+"/images.json", 'w') as f:
            json.dump(urls, f)
        print(len(urls), 'image metadata saved in', args.data + "/images.json")
    else:
        with open(args.data + "/images.json", 'r') as f:
            urls = json.load(f)
        print(len(urls), 'image metadata loaded in', args.data+"/urls.json")

    if args.download:
        img_dir = os.path.join(args.data, 'img')
        print("Downloading images to", img_dir)
        if args.single_thread_dl:
            cnt = 0
            for i in urls:
                downloadMedia(i, root=img_dir, random_subsets={'train':0.95, 'val':0.05})
                cnt += 1
                if cnt % 500 == 0:
                    print('\timage', cnt,'on', len(urls), 'downloaded')
        else:
            gbif_dl.io.download(urls, root=img_dir, random_subsets={'train':0.95, 'val':0.05},
                            nb_workers=6, retries=1, verbose=args.verbose)

        print('Images downloaded')
    else:
        print('Download skipped')


if __name__=="__main__":
    main()
