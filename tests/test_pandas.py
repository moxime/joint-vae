import pandas as pd
import numpy as np
from numpy.random import choice as choose


species = ['dog', 'cat']
colors = ['black', 'red']

habitats = ['city', 'country']

N = 40
animals = []
for i in range(N // 2):

    animals.append({'specy': choose(species),
                    'color': choose(colors),
                    'habitat': choose(habitats),
                    'size': {'legs': np.random.rand(),
                            'head': np.random.rand()}})

    animals.append({'specy': choose(species),
                    'color': choose(colors),
                    'habitat': choose(habitats),
                    'size': {'paw': np.random.rand()}})

    
df = pd.DataFrame.from_records(animals, columns=('specy', 'color', 'habitat', 'size'))
    

df2 = df.drop('size', axis=1).join(pd.DataFrame(df['size'].values.tolist()))

df2.set_index(['specy', 'color', 'habitat'], inplace=True)

print(df2)
df3 = df2.reorder_levels(['color', 'habitat', 'specy']).stack()

print(df3)

df4 = df2.reset_index(['specy', 'habitat'])


# pd.set_option('max_colwidth', 10)
# print(df2.to_string())
