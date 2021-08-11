import argparse
import json
import os
import requests
import hashlib

gbif = 'https://api.gbif.org/v1'

def main():
    parser = argparse.ArgumentParser(description='Cos4Cloud use gbif api to retrieve images records from gbif species ids')
    parser.add_argument('-d', '--data',
                        help='output directory', required=True)
    parser.add_argument('-s', '--species', type=argparse.FileType('r'),
                        help='json file listing providers ids', required=True)
    parser.add_argument('-v', '--verbose', action='store_true', default=False)
    parser.add_argument('-n', '--number', type=int, metavar='N', default=1000,
                        help='Number of images per species')

    args = parser.parse_args()

    os.makedirs(args.data, exist_ok=True)
    species = json.load(args.species)

    not_illustrated = []
    urls = []
    publishers = {}
    licenses = {}
    rightsHolders = {}
    species_count = 0
    print('crawling gbif for illustrations on', len(species),'species')
    for s in species:
        count = 0
        offset = 0
        limit = args.number if args.number<50 else 50
        cont = True
        while cont:
            doc = {}
            url = gbif+'/occurrence/search?taxonKey='+str(s)+'&mediaType=stillImage&offset='+str(offset)+'&limit='+str(limit)
            try:
                resp = requests.get(url)
                if resp.status_code == 200:
                    doc = resp.json()
                    cont = not doc.get('endOfRecords', True)

            except Exception as e:
                print('Error while retrieving url', url)
                print(e)
                cont = False
            results = doc.get('results', [])
            for r in results:
                label = r.get('acceptedTaxonKey', None)
                if label:
                    media = r.get('media', [])
                    for m in media:
                        publisher = m.get('publisher', None)
                        license = m.get('license', None)
                        rightsHolder = m.get('rightsHolder', m.get('creator', None))
                        url = m.get('identifier', None)
                        if url and license and rightsHolder: #and publisher ?
                            basename = hashlib.sha1(url.encode("utf-8")).hexdigest()
                            license_idx = licenses.get(license, len(licenses))
                            licenses[license]=license_idx
                            rightsHolder_idx = rightsHolders.get(rightsHolder, len(rightsHolders))
                            rightsHolders[rightsHolder]=rightsHolder_idx
                            publisher_idx = publishers.get(publisher, len(publishers))
                            publishers[publisher]=publisher_idx
                            urls.append({
                                'url': url,
                                'basename': basename,
                                'label': label,
                                'publisher': publisher_idx,
                                'license': license_idx,
                                'rightsHolder': rightsHolder_idx
                            })
                            count += 1
                        # else:
                        #     print(m)
            if count>args.number:
                cont = False
            else:
                offset += limit
        species_count += 1
        print('\tspecies',s,'('+str(species_count)+'/'+str(len(species))+')', 'is illustrated by',count,'images')
        if not count:
            not_illustrated.append(s)
    with open(args.data+'/urls.json', 'w') as f:
        json.dump(urls, f)
    with open(args.data+'/images.json', 'w') as f:
        json.dump({
            'publisher': list(publishers),
            'license': list(licenses),
            'rightsHolder': list(rightsHolders),
            'urls': urls
        }, f)
    print('Saved', len(urls), 'image records')
    if len(not_illustrated):
        print(len(not_illustrated),'species were not illustrated, see',args.data+'/not_illustrated.json',
              'for the details')
        with open(args.data+'/not_illustrated.json', 'w') as f:
            json.dump(not_illustrated, f)

if __name__=="__main__":
    main()