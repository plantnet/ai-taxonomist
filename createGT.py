import argparse
import json
import os
import pygbif


def main():
    parser = argparse.ArgumentParser(description='build c4c-identify ground truth')
    parser.add_argument('-d', '--data',
                        help='output directory', required=True)
    parser.add_argument('--images', default=None, help='list of images urls')
    parser.add_argument('--name-type', type=str, choices=['canonicalName', 'scientificName'], default='canonicalName',
                        help='Use canonical or scientific names')
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    if not args.images:
        args.images = os.path.join(args.data, 'images.json')

    gt_path = os.path.join(args.data, 'GT')
    os.makedirs(gt_path, exist_ok=True)

    id2class = {}
    species_id = []
    print('=> loading network mapping')
    with open(os.path.join(args.data,'network', 'mapping.json')) as f:
        id2class = json.load(f)

    print('=> retrieving names from gbif and building class mapping')
    mapping = [{} for i in range(len(id2class))]
    for k,v in id2class.items():
        name_usage = pygbif.species.name_usage(key=k, data='name')
        mapping[v] = {'name': name_usage.get(args.name_type, None), 'species_id':k}
    with open(os.path.join(gt_path, 'classes.json'), 'w') as f:
        json.dump(mapping, f)

    print('=> loading images urls and building images mapping')
    images = dict()
    urls = list()
    with open(args.images) as f:
        urls = json.load(f)
    for i in urls:
        basename = i.get('basename', None)
        species_id = i.get('label', None)
        if basename and species_id:
            class_id = id2class.get(species_id, -1)
            if class_id>0:
                images[basename] = i
                images[basename]['class_id'] = class_id
    with open(os.path.join(gt_path, 'images.json'), 'w') as f:
        json.dump(images, f)


if __name__=="__main__":
    main()
