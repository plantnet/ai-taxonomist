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
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    if not args.species_id:
        args.species_id = args.data+"/species_id.json"

    gt = {}
    cnt = 0
    with open(args.species_id, 'r') as f:
        data = json.load(f)
        for s in data:
            resp = requests.get(gbif + '/species/' + str(s) + '/name')
            if resp.status_code == 200:
                data = resp.json()
                name = data.get('canonicalName', None)
                gt[s] = name
                cnt += 1
                if args.verbose and cnt % 50 == 0:
                    print('Retrieved', cnt, 'names')
            else:
                print('Unable to get gbif info for species', s)
    with open(args.data+'/gt_id2name.json', 'w') as f:
        json.dump(gt, f)

if __name__=="__main__":
    main()