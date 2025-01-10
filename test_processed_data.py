from pathlib import Path

import torch


def main():
    # load the preprocessed data
    db_name = 'IMDB-BINARYFeatures'
    path = Path(f'DATA/{db_name}')
    # load the graph data with torch
    graph_data = torch.load(path.joinpath(f'processed/{db_name}.pt'))
    pass

if __name__ == '__main__':
    main()