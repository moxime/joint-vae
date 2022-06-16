import shutil, os

directory = 'jobs/out'


for fn in os.scandir(directory):
    if fn.name.endswith('.out'):
        
        filename = fn.path
        old_filename = filename + '.old'
        shutil.copy(filename, old_filename)

        with open(old_filename, 'rb') as f:
            with open(filename, 'w') as f_:
                firsts = set()
                lasts = set()
                penultians = set()
                for i, l in enumerate(f.readlines()):
                    f_.write(l.split(b'\r')[-1].decode())

