#
# Copyright (C)  2020  University of Pisa
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
import argparse

from datasets import *

DATASETS = {
    'REDDIT-BINARY': RedditBinary,
    'REDDIT-MULTI-5K': Reddit5K,
    'COLLAB': Collab,
    'ENZYMES': Enzymes,
    'PROTEINS': Proteins,
    'DD': DD,
    'IMDB-BINARY': IMDBBinary,
    'IMDB-BINARYFeatures': IMDBBinaryFeatures,
    'IMDB-MULTI': IMDBMulti,
    'IMDB-MULTIFeatures': IMDBMultiFeatures,
    'NCI1': NCI1,
    'NCI1Features': NCI1Features,
    'DHFR': DHFR,
    'DHFRFeatures': DHFRFeatures,
    'Mutagenicity': Mutagenicity,
    'MutagenicityFeatures': MutagenicityFeatures,
    'NCI109': NCI109,
    'NCI109Features': NCI109Features,
    'Snowflakes': Snowflakes,
    'SnowflakesFeatures': SnowflakesFeatures,
    'CSL': CSL,
    'CSLFeatures': CSLFeatures,
}

EXPERIMENT_DATASETS = {
    'IMDB-BINARY': IMDBBinary,
    'IMDB-BINARYFeatures': IMDBBinaryFeatures,
    'IMDB-MULTI': IMDBMulti,
    'IMDB-MULTIFeatures': IMDBMultiFeatures,
    'NCI1': NCI1,
    'NCI1Features': NCI1Features,
    'DHFR': DHFR,
    'DHFRFeatures': DHFRFeatures,
    'Mutagenicity': Mutagenicity,
    'MutagenicityFeatures': MutagenicityFeatures,
    'NCI109': NCI109,
    'NCI109Features': NCI109Features,
    'Snowflakes': Snowflakes,
    'SnowflakesFeatures': SnowflakesFeatures,
    'CSL': CSL,
    'CSLFeatures': CSLFeatures,
}


def get_args_dict():
    parser = argparse.ArgumentParser()

    parser.add_argument('DATA_DIR',
                        help='where to save the datasets')
    parser.add_argument('--dataset-name', dest='dataset_name',
                        choices=EXPERIMENT_DATASETS.keys(), default='all', help='dataset name [Default: \'all\']')
    parser.add_argument('--outer-k', dest='outer_k', type=int,
                        default=10, help='evaluation folds [Default: 10]')
    parser.add_argument('--inner-k', dest='inner_k', type=int,
                        default=None, help='model selection folds [Default: None]')
    parser.add_argument('--use-one', action='store_true',
                        default=False, help='use 1 as feature')
    parser.add_argument('--use-degree', dest='use_node_degree', action='store_true',
                        default=False, help='use degree as feature')
    parser.add_argument('--no-kron', dest='precompute_kron_indices', action='store_false',
                        default=True, help='don\'t precompute kron reductions')

    return vars(parser.parse_args())


def preprocess_dataset(name, args_dict):
    dataset_class = EXPERIMENT_DATASETS[name]
    if name == 'ENZYMES':
        args_dict.update(use_node_attrs=True)
    if 'Features' in name:
        args_dict.update(use_node_attrs=True)
    if name == 'CSL':
        args_dict.update(outer_k=5)
    dataset_class(**args_dict)


if __name__ == "__main__":
    args_dict = get_args_dict()

    print(args_dict)

    dataset_name = args_dict.pop('dataset_name')
    if dataset_name == 'all':
        for name in EXPERIMENT_DATASETS:
            print(f'Preprocessing {name}...')
            preprocess_dataset(name, args_dict)
    else:
        print(f'Preprocessing {dataset_name}...')
        preprocess_dataset(dataset_name, args_dict)