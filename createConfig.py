import argparse
import json
import os
import sys
import glob

def main():
    parser = argparse.ArgumentParser(description='Cos4Cloud use gbif api to convert canonical names to gbif species ids')
    parser.add_argument('-d', '--data',
                        help='output directory', required=True)
    offline = parser.add_argument_group('offline', 'parameters who impact snoop preparation')
    offline.add_argument('--memory', type=int, default=48000, help='Memory (in Mb) to use while building the image index')
    offline.add_argument('--threads', type=int, default=24, help='Number of threads to use while building the image index')
    online = parser.add_argument_group('online', 'parameters who impact snoop at run time')
    online.add_argument('--gpu', type=int, default=[], nargs='*', help='Wich gpu to use at run time, if any')
    online.add_argument('--port', type=int, default=8080, help='which port to launch the snoop on')
    online.add_argument('--snoop-threads', type=int, default=10, help='how many parallel request to handle')
    online.add_argument('--proxy', type=str, default="", help='proxy to use while downloading the images')

    args = parser.parse_args()

    print('=> loading network mapping')
    with open(os.path.join(args.data, 'network', 'network.json')) as f:
        network = json.load(f)

    print('=> instantiating templates')
    compression = network.get('feat_size', 1024)//2
    crop_size = network.get('img_size', 224)
    tile_size = int(crop_size * 1.1 + 0.5)
    values = {
        # Image Index
        '@COMPRESSION@': compression,
        '@EMBEDDER_SAMPLES@': compression*100,
        '@MEMORY@': args.memory,
        '@THREADS@': args.threads,
        # Descriptors
        '@GPU@': 'true' if len(args.gpu) else 'false',
        '@TILE_SIZE@': tile_size,
        '@CROP_SIZE@': crop_size,
        '@NETWORK@': network.get('network', os.path.join('network','network.pt')),
        '@NUM_CLASSES@': network.get('num_classes', 1000),
        '@FEATURES@': network.get('features', 'features'),
        '@FEAT_SIZE@': network.get('feat_size', 1024),
        '@LOGITS@': network.get('logits', 'logits'),
        # Run time
        '@PORT@': network.get('port', 8080),
        '@MAIN_THREAD@': args.snoop_threads,
        '@HTTP_PROXY@': args.proxy,
        '@CNN_VERSION@': network.get('arch', 'dnn'),
        '@GPU_ID@': args.gpu
    }

    config_dir = os.path.join(args.data, 'Config')
    os.makedirs(config_dir, exist_ok=True)
    template_dir = os.path.join(os.path.dirname(sys.argv[0]), 'Config')

    for tpl in glob.glob(os.path.join(template_dir, '*.in')):
        cfg = os.path.join(config_dir, os.path.basename(tpl)[:-3])
        with open(tpl, 'r') as f:
            with open(cfg, 'w') as out:
                for line in f:
                    for k,v in values.items():
                        line = line.replace(k,str(v))
                    out.write(line)


if __name__=="__main__":
    main()