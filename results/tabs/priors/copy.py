import os
import shutil
from os.path import join
subdir = 'results/tabs/priors'

l = os.listdir(subdir)

for f in [_ for _ in l if _.endswith('.ini')]:

    f_ = os.path.splitext(f)[0] + '-a-4-1' + '.ini'

    print(f_)

    shutil.copyfile(join(subdir, f), join(subdir, f_))
