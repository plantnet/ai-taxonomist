import os
import filetype

data = '/data/jcl/frogs'

for (dirpath, dirnames, filenames) in os.walk(data+'/img'):
    for f in filenames:
        filepath = os.path.join(dirpath,f)
        ext = filetype.guess_extension(filepath)
        if len(ext)>0:
            os.rename(filepath, filepath+'.'+ext)
        else:
            print(dirpath, f,ext)