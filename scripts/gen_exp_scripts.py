#!/usr/bin/env python
# coding: utf-8
import os

with open('/home/k/k202141/rootgit/AI4EO-MapYourCity/scripts/header.txt') as f:
    header = f.readlines()

with open('/home/k/k202141/rootgit/AI4EO-MapYourCity/scripts/expsource.txt') as f:
    explines = f.readlines()

sroot = '/home/k/k202141/rootgit/AI4EO-MapYourCity/scripts/'

with open(os.path.join(sroot, 'batch.sh'), 'w') as ff:
    for i, l in enumerate(explines):
        sf = f'submit_{i:02d}.sh'
    
        with open(os.path.join(sroot, sf), 'w') as f:
            for h in header:
                f.write(h)
            f.write(l)
            print(l)
    
            f.close()
        ff.write(f'sbatch {sf}\n')
        ff.write('sleep 100\n')
    ff.close()
