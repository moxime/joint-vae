import pandas as pd


def process_csv(csv_file, header):

    df = pd.read_csv(csv_file, header=header)
