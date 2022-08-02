import os
import random

val_percent = 0.05
data = '/data/jcl/frogs'

# crawl dir
for d in os.listdir(data+'/img/vrac'):
    files = []
    for f in os.listdir(data+'/img/vrac/'+d):
        files.append(f)
    if not len(files) == 0:
        random.shuffle(files)
        n = len(files)
        v = int(n * val_percent + 0.5)
        if v>0:
            os.makedirs(data+'/img/val/'+d, exist_ok=True)
            for f in files[:v]:
                os.rename(data+'/img/vrac/'+d+'/'+f, data+'/img/val/'+d+'/'+f)
        os.makedirs(data+'/img/train/'+d, exist_ok=True)
        for f in files[v:]:
            os.rename(data + '/img/vrac/' + d + '/' + f, data + '/img/train/' + d + '/' + f)
