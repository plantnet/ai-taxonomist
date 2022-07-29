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


def crawlNames(args: dict, temp_data: str) -> list:
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
        output = os.path.join(temp_data, 'no_species.json')
        print('some species names do not match any species, see', output, 'for details')
        with open(output, 'w') as f:
            json.dump(other, f)

    with open(os.path.join(temp_data, 'species_ids.json'), 'w') as f:
        json.dump(species_ids, f)

    return species_ids


def downloadMedia(item: MediaData, root: str, random_subsets: dict = {'train': 0.95, 'val': 0.05},
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
        proxies = {'http': proxy, 'https': proxy}
    res = None
    try:
        res = requests.get(url, proxies=proxies)
    except Exception as e:
        print('Failed to download', url, ':', e)
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
    parser.add_argument('--workers', type=int, metavar='N', default=10, help='number of // downloads')
    parser.add_argument('--no-fix', action='store_false', dest='fix',
                        help='do not fix train/val split. Expect troubles !')
    parser.add_argument('--check-images', action='store_true', dest='check',
                        help='check that images are actual images, takes a while but prevents troubles during train')

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

    train_data = args.data
    temp_data = os.path.join(args.data, 'temporary')
    for d in [train_data, temp_data]:
        os.makedirs(d, exist_ok=True)

    species_ids = []
    images_file = os.path.join(temp_data, "images.json")
    if args.crawl:
        print('Building data generator')
        if args.doi:
            # use a precomputed gbif query from its doi
            if args.names or args.species:
                print('it does not make sense to provide both --doi and --name or --species arguments')
                sys.exit(-1)
            # save doi for online usage
            doi = {'gbif_doi': args.doi}
            with open(os.path.join(temp_data, 'doi.json'), 'w') as f:
                json.dump(doi, f)

            data_generator = gbif_dl.dwca.generate_urls(args.doi,
                                                        dwca_root_path=os.path.join(temp_data, "dwcas"),
                                                        label="speciesKey",
                                                        mediatype='StillImage',
                                                        one_media_per_occurrence=False, )
        else:
            # retrieve media from a name or a species list and an optional providers list
            if args.names:
                if args.species:
                    print('it does not make sens to provide both --name and --species arguments')
                    sys.exit(-1)
                species_ids = crawlNames(args, temp_data)
            elif args.species:
                species_ids = json.load(args.species)
            else:
                print('no doi, no species, no names => nothing to do!')
                sys.exit(-1)

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

        print('Retrieving image metadata...')
        urls = []
        for i in data_generator:
            urls.append(i)
            if len(urls) % 500 == 0:
                print(len(urls), 'image metadata retrieved')
        with open(images_file, 'w') as f:
            json.dump(urls, f)
        print(len(urls), 'image metadata saved in', images_file)
    else:
        print('Crawl skipped, loading stored metadata')
        with open(images_file, 'r') as f:
            urls = json.load(f)
        print(len(urls), 'image metadata loaded from', images_file)

    img_dir = os.path.join(train_data, 'img')
    if args.download:
        print("Downloading images to", img_dir)
        if args.single_thread_dl:
            cnt = 0
            for i in urls:
                downloadMedia(i, root=img_dir, random_subsets={'train': 0.95, 'val': 0.05})
                cnt += 1
                if cnt % 500 == 0:
                    print('\timage', cnt, 'on', len(urls), 'downloaded')
        else:
            gbif_dl.io.download(urls, root=img_dir, random_subsets={'train': 0.95, 'val': 0.05},
                                nb_workers=args.workers, retries=1, loglevel="INFO")

        print('Images downloaded')
    else:
        print('Download skipped')

    if args.check:
        from PIL import Image

    if args.fix:
        # remove empty image files
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                path = os.path.join(root, f)
                if os.path.getsize(path) == 0:
                    os.remove(path)
                    print('Removed empty file', path)
                elif args.check:
                    try:
                        img = Image.open(path)
                        img.load()
                    except:
                        os.remove(path)
                        print('Removed buggy file', path)

        # remove empty train/val directory
        for sub in ['train', 'val']:
            for d in os.listdir(os.path.join(img_dir, sub)):
                try:
                    dd = os.path.join(img_dir, sub, d)
                    os.rmdir(dd)
                    print('Removed empty', sub, 'class', d)
                except OSError:
                    pass
        # avoid to have some classes illustrated only in the validation set
        train_dir = os.path.join(img_dir, 'train')
        train_classes = os.listdir(train_dir)
        val_dir = os.path.join(img_dir, 'val')
        val_classes = os.listdir(val_dir)
        for c in val_classes:
            if not c in train_classes:
                print('class', c, 'is only in val, moving it to train')
                os.rename(os.path.join(val_dir, c), os.path.join(train_dir, c))
    else:
        print('train/val set fix skipped')


if __name__ == "__main__":
    main()
