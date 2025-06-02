import pandas as pd


def process_csv(csv_file, header=2, index_col=1):

    df = pd.read_csv(csv_file, header=[*range(header)], index_col=[*range(index_col)])

    i_names = set(df.columns.names) | set(df.index.names)

    col_names = ['set', 'method', 'measures']
    assert set(col_names) <= i_names
    assert set(df.columns.names) <= set(col_names)

    for _ in df.index.names:
        if _ in col_names:
            df = df.unstack(_)

    df = df.reorder_levels(col_names, axis='columns')
    return df


if __name__ == '__main__':

    csv_file = ('/tmp/oopenood.csv', 2, 2)
    csv_file = ('/tmp/ib.csv', 3, 1)

    df = process_csv(*csv_file)

    print(df.index.names)
    print(df.columns.names)
