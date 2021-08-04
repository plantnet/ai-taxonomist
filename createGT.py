import argparse
import json
import os
import requests

gbif = 'https://api.gbif.org/v1'

def main():
    parser = argparse.ArgumentParser(description='Cos4Cloud use gbif api to convert canonical names to gbif species ids')
    parser.add_argument('-d', '--data',
                        help='output directory', required=True)
    parser.add_argument('-i', '--species_id', help='json file containing the species id list',
                        default = None, required=False)
    parser.add_argument('--urls', default=None, help='list of images urls')
    parser.add_argument('--id2name', help='stored id2name dict', default=None, required=False)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    if not args.species_id:
        args.species_id = os.path.join(args.data, 'species_id.json')

    if not args.urls:
        args.urls = os.path.join(args.data, 'urls.json')

    gt_path = os.path.join(args.data, 'GT')
    os.makedirs(gt_path, exist_ok=True)
    id2name = {}
    cnt = 0
    if not args.id2name:
        print('=> querying gbif to retrieve canonical names for {}'.format(args.species_id))
        with open(args.species_id, 'r') as f:
            data = json.load(f)
            for s in data:
                resp = requests.get(gbif + '/species/' + str(s) + '/name')
                if resp.status_code == 200:
                    data = resp.json()
                    name = data.get('canonicalName', None)
                    id2name[s] = name
                    cnt += 1
                    if args.verbose and cnt % 50 == 0:
                        print('Retrieved', cnt, 'names')
                else:
                    print('Unable to get gbif info for species', s)
        with open(os.path.join(gt_path,'tmp','gt_id2name.json'), 'w') as f:
            json.dump(id2name, f)
    else:
        print('=> loading canonical names from {}'.format(args.id2name))
        with open(args.id2name) as f:
            id2name = json.load(f)

    print('=> loading network mapping')
    with open(os.path.join(args.data,'network', 'mapping.json')) as f:
        id2class = json.load(f)

    print('=> building class mapping')
    mapping = [{} for i in range(len(id2class))]
    for k,v in id2class.items():
        mapping[v] = {'name': id2name.get(k, None), 'species_id':k}
    if args.verbose:
        invMapping = dict()
        notFound = []
        for k,v in id2name.items():
            invMapping[v] = id2class.get(k, -1)
            if invMapping[v] == -1:
                notFound.append({k:v})
        if len(notFound):
            print('The network only learned', len(mapping), 'classes on', len(invMapping), 'given species ids')
    with open(os.path.join(gt_path, 'classes.json'), 'w') as f:
        json.dump(mapping, f)

    print('=> loading images urls')
    with open(args.urls) as f:
        tmp = json.load(f)
    urls = dict()
    for i in tmp:
        urls[i.get('basename', '')] = i.get('url', '')

    print('=> build image mapping')
    img_dir = os.path.join(args.data, 'img')
    images = dict()
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        for f in filenames:
            species_id = dirpath.split('/')[-1]
            class_id = id2class.get(species_id, -1)
            basename = os.path.splitext(f)[0]
            url = urls.get(basename, '')
            images[f] = {'class_id': class_id, 'url':url} # should do something about licenses !

    with open(os.path.join(gt_path, 'images.json'), 'w') as f:
        json.dump(images, f)


if __name__=="__main__":
    main()