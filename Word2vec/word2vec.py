# 主要探讨的是关于skip-Gram的word2vec
import collections
import math
import os
import random
import zipfile
import numpy as np   
import urllib
import tensorflow as tf  

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print ('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception(
            'Failed to vertify' + filename +' .can you get to it with a browser?'
        )
    return filename

filename = maybe_download('text8.zip', 31344016)
