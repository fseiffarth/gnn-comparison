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
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, global_max_pool, GATv2Conv


class GATv2(nn.Module):
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        num_layers = config['num_layers']
        dim_embedding = config['dim_embedding']
        self.heads = config['heads']
        self.aggregation = config['aggregation']  # can be mean or max
        self.dropout = config['dropout']

        if self.aggregation == 'max':
            self.fc_max = nn.Linear(self.heads * dim_embedding, self.heads * dim_embedding)

        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            dim_input = dim_features if i == 0 else dim_embedding

            # Overwrite aggregation method (default is set to mean
            conv = GATv2Conv(in_channels=dim_input, out_channels=dim_embedding, heads=self.heads, aggr=self.aggregation,
                           dropout=self.dropout)
            self.layers.append(conv)
            # if heads > 1 add a linear layer to reduce the dimension
            if self.heads > 1:
                self.layers.append(nn.Linear(self.heads * dim_embedding, dim_embedding))

        # For graph classification
        self.fc1 = nn.Linear(dim_embedding, dim_embedding)
        self.fc2 = nn.Linear(dim_embedding, dim_target)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, layer in enumerate(self.layers):
            if self.heads > 1:
                x = self.layers[2 * i](x, edge_index)
                if self.aggregation == 'max':
                    x = torch.relu(self.fc_max(x))
                x = self.layers[2 * i + 1](x)
                if i == len(self.layers) // 2 - 1:
                    break
            else:
                x = layer(x, edge_index)
                if self.aggregation == 'max':
                    x = torch.relu(self.fc_max(x))

        x = global_max_pool(x, batch)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
