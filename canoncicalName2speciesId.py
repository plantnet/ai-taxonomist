import argparse
import json
import os
import requests

gbif = 'https://api.gbif.org/v1'

def main():
    parser = argparse.ArgumentParser(description='Cos4Cloud use gbif api to convert canonical names to gbif species ids')
    parser.add_argument('-d', '--data',
                        help='output directory', required=True)
    parser.add_argument('-n', '--names', type=argparse.FileType('r'),
                        help='json file listing providers ids', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)

    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)
    species = []
    other = []
    cnt = 0
    for n in args.names:
        n = n.rstrip()
        resp = requests.get(gbif+'/species/match?name='+n)
        if resp.status_code == 200:
            data = resp.json()
            status = data.get('status', 'unknown')
            match = data.get('matchType', 'unknown')
            rank = data.get('rank', None)
            id = data.get('speciesKey', None)

            if match == 'EXACT' and rank == 'SPECIES' and id:
                species.append(id)
            else:
                print(n, 'does not match any species in gbif', {'name':n, 'rank':rank, 'status':status, 'match':match})
                if args.verbose:
                    print(data)
                other.append({'name':n, 'gbif':data})
            cnt += 1
            if args.verbose and cnt % 50 == 0:
                print('Translated', cnt , 'species')
        else:
            print('Unable to get gbif info for name', n)

    # remove duplicates
    if args.verbose:
        print('Before remove duplicate, got', len(species), 'species id')
    species = set(species)
    species = list(species)
    if args.verbose:
        print('After remove duplicate, got', len(species), 'species id')
    with open(args.data + '/species_id.json', 'w') as f:
        json.dump(species, f)
    if len(other)>0:
        with open(args.data + '/no_species.json', 'w') as f:
            json.dump(other, f, indent=4)

    print('Saved', len(species), 'species id. Got', len(other), 'errors');
    if len(other)>0:
        print('check file other.json')

if __name__=="__main__":
    main()