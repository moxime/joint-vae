import pandas as pd
import numpy as np
from numpy.random import choice as choose


species = ['dog', 'cat']
colors = ['black', 'red']

habitats = ['city', 'country']

N = 10

animals = []
for i in range(N):

    animals.append({'specy': choose(species),
                    'color': choose(colors),
                    'habitat': choose(habitats),
                    'siz': {'legs': np.random.rand(),
                            'head': np.random.rand()}})

    animals.append({'specy': choose(species),
                    'color': choose(colors),
                    'habitat': choose(habitats),
                    'siz': {'paw': np.random.rand()}})


    
df = pd.DataFrame.from_records(animals, columns=('specy', 'color', 'habitat', 'siz'))
    

df2 = df.drop('siz', axis=1).join(pd.DataFrame(df.siz.values.tolist()))

df2.style.apply('font-weight: bold')

print(df2)

pd.set_option('max_colwidth', 5)
print(df2.to_string())
