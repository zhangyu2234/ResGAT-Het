import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_softmax

class ResGATHetLayer(MessagePassing):
    def __init__(self, in_features, out_features, num_heads, dropout_rate=0.5):
        super(ResGATHetLayer, self).__init__(aggr='add')
        self.num_heads = num_heads
        self.out_features_per_head = out_features // num_heads
        self.self_attention_lin = nn.Linear(self.out_features_per_head, self.out_features_per_head)
        self.lin = nn.Linear(in_features, self.out_features_per_head * num_heads)
        self.att_lin = nn.Linear(3 * self.out_features_per_head, 1)
        # Additional linear layer for residual connection
        self.residual_lin = nn.Linear(in_features, self.out_features_per_head * num_heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_weight=None):
        # Apply the residual connection transformation
        residual = self.residual_lin(x)

        x = self.lin(x)
        x = self.dropout(x)
        x = x.view(-1, self.num_heads, self.out_features_per_head)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), device=x.device)

        x_list = []
        for head in range(self.num_heads):
            x_head = x[:, head, :]
            x_i = x_head[edge_index[1]]
            repeated_x = x_head.index_select(0, edge_index[0])
            alpha_self = self.self_attention_lin(x_head).index_select(0, edge_index[0])
            alpha_input = torch.cat([repeated_x, x_i, alpha_self], dim=-1)
            alpha = F.leaky_relu(self.att_lin(alpha_input), negative_slope=0.2)
            alpha = scatter_softmax(alpha, edge_index[0], dim=0)
            edge_weight_head = edge_weight * alpha.view(-1)
            x_head_out = self.propagate(edge_index, x=x_head, edge_weight=edge_weight_head)
            # Add the residual connection here
            x_head_out += residual.view(-1, self.num_heads, self.out_features_per_head)[:, head, :]
            x_list.append(x_head_out)

        x_out = torch.cat(x_list, dim=1)
        return x_out


class ResGATHet(nn.Module):
    def __init__(self, in_features_dict, out_features, num_edge_types, num_heads, dropout_rate=0.2):
        super(ResGATHet, self).__init__()

        # Embedding layers for each node type
        self.embedding_layers = nn.ModuleDict({
            node_type: nn.Linear(in_feat, out_features)
            for node_type, in_feat in in_features_dict.items()
        })

        self.att_layers = nn.ModuleList(
            [ResGATHetLayer(out_features, out_features, num_heads) for _ in range(num_edge_types)]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.final_transform = nn.Linear(num_edge_types * out_features, out_features)
        self.out_features = out_features

    def forward(self, data):
        # If the data is a Tensor
        if isinstance(data, torch.Tensor):
            # Process the Tensor using the embedding layer associated with 'user'
            x = self.embedding_layers['user'](data)
            #...Other code that needs to process individual Tensors...
            return x, ['user']

        # Otherwise, process the Data object
        node_types = [key for key, value in data._store.items() if 'x' in value]
        edge_types = [key for key, value in data._store.items() if 'edge_index' in value]

        # Concatenate node features for all types
        x = torch.cat([self.embedding_layers[node_type](getattr(data, node_type).x) for node_type in node_types], dim=0)

        # Aggregate features across different relation types
        x_list = []
        for i, rel_type in enumerate(edge_types):
            edge_index = getattr(data, rel_type).edge_index

            # Check if edge weights exist for this relation type, if not, assign default weights
            if hasattr(getattr(data, rel_type), 'edge_attr'):
                edge_weight = getattr(data, rel_type).edge_attr
            else:
                edge_weight = torch.ones((edge_index.size(1),), device=x.device)

            x_rel = self.att_layers[i](x, edge_index, edge_weight)
            x_list.append(x_rel)

        x_combined = torch.cat(x_list, dim=1)
        x_combined = self.dropout(x_combined)
        x = self.final_transform(x_combined)
        x = torch.stack(x_list, dim=1)
        x = x.view(-1, self.out_features)
        start_idx = 0
        node_type_ranges = {}
        for node_type in node_types:
          num_nodes = getattr(data, node_type).x.shape[0]
          node_type_ranges[node_type] = (start_idx, start_idx + num_nodes)
          start_idx += num_nodes

        return x, node_type_ranges