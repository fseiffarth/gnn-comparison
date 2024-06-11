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

import torch

from EndToEnd_Evaluation import main as endtoend


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', dest='config_file')
    parser.add_argument('--experiment', dest='experiment', default='endtoend')
    parser.add_argument('--result-folder', dest='result_folder', default='RESULTS')
    parser.add_argument('--dataset-name', dest='dataset_name', default='none')
    parser.add_argument('--outer-folds', dest='outer_folds', default=10)
    parser.add_argument('--outer-processes', dest='outer_processes', default=2)
    parser.add_argument('--inner-folds', dest='inner_folds', default=5)
    parser.add_argument('--inner-processes', dest='inner_processes', default=1)
    parser.add_argument('--debug', action="store_true", dest='debug')
    return parser.parse_args()

# local args python Launch_Experiments.py --config-file config_DGCNN.yml --result-folder RESULTS --outer-processes 10 --inner-processes 3 --dataset-name CSL
# remote args python Launch_Experiments.py --config-file config_DGCNN.yml --result-folder /home/mlai21/seiffart/RESULTS/GNNComparison --outer-processes 3 --inner-processes 10



if __name__ == "__main__":
    args = get_args()
    # export OMP_NUM_THREADS=1
    torch.set_num_threads(1)

    if args.dataset_name != 'none':
        datasets = [args.dataset_name]
    if args.dataset_name == 'all':
        datasets = ['Mutagenicity', 'NCI109', 'DHFR', 'NCI1', 'IMDB-BINARY', 'IMDB-MULTI']
    if args.dataset_name == 'features':
        datasets = ['DHFRFeatures', 'IMDB-BINARYFeatures', 'IMDB-MULTIFeatures', 'MutagenicityFeatures', 'NCI1Features', 'NCI109Features']
    if args.dataset_name == 'synthetic':
        datasets = ['Snowflakes', 'EvenOddRingsCount16', 'EvenOddRings2_16',  'LongRings100']
    if args.dataset_name == 'CSL':
        datasets = ['CSL']
        # set outer folds to 5
        args.outer_folds = 5
        # bound outer processes to 5
        args.outer_processes = max(5, int(args.outer_processes))
    if datasets is None:
        # raise error with help
        raise ValueError('dataset-name not recognized\n'
                            'please choose from: all, features, missing, synthetic\n'
                         'or specify a single dataset name')


    # get device
    config_file = args.config_file
    experiment = args.experiment

    for dataset_name in datasets:
        try:
            endtoend(config_file, dataset_name,
                     outer_k=int(args.outer_folds), outer_processes=int(args.outer_processes),
                     inner_k=int(args.inner_folds), inner_processes=int(args.inner_processes),
                     result_folder=args.result_folder, debug=args.debug)
        
        except Exception as e:
            raise e  # print(e)