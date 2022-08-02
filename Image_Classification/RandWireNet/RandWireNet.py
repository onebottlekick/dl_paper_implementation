import os

import networkx as nx
import torch
import torch.nn as nn
import yaml


class RandomGraph:
    def __init__(self, num_nodes, graph_probability, nearest_neighbour_k=4):
        self.num_nodes = num_nodes
        self.graph_probability = graph_probability
        self.nearest_neighbor_k = nearest_neighbour_k
        
    def make_graph(self):
        graph = nx.random_graphs.connected_watts_strogatz_graph(self.num_nodes, self.nearest_neighbor_k, self.graph_probability)
        return graph
    
    def get_graph_config(self, graph):
        incoming_edges = {}
        incoming_edges[0] = []
        nodes = [0]
        last = []
        
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors.sort()
            
            edges = []
            passes = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor + 1)
                    passes.append(neighbor)

            if not edges:
                edges.append(0)
            
            incoming_edges[node + 1] = edges
            if passes == neighbors:
                last.append(node + 1)
                
            nodes.append(node + 1)
        
        incoming_edges[self.num_nodes + 1] = last
        nodes.append(self.num_nodes + 1)
        
        return nodes, incoming_edges
    
    def save_graph(self, graph, path):
        os.makedirs('graph', exist_ok=True)
        with open(os.path.join('graph', path), 'w') as f:
            yaml.dump(graph, f)
            
    def load_graph(self, path):
        with open(os.path.join('graph', path), 'r') as f:
            return yaml.load(f, Loader=yaml.Loader)
        
        
class SeperatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(SeperatedConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=bias)
        
    def forward(self, x):
        return self.pointwise(self.conv(x))
    

class UnitLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super(UnitLayer, self).__init__()

        self.unit = nn.Sequential(
            nn.ReLU(),
            SeperatedConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return self.unit(x)

        
class GraphNode(nn.Module):
    def __init__(self, input_degree, in_channels, out_channels, stride=1):
        super(GraphNode, self).__init__()
        
        self.input_degree = input_degree
        
        if len(input_degree) > 1:
            self.params = nn.Parameter(torch.ones(len(input_degree), requires_grad=True))
        
        self.unit = UnitLayer(in_channels, out_channels, stride=stride)
        
    def forward(self, *inputs):
        if len(self.input_degree) > 1:
            operation = (inputs[0]*torch.sigmoid(self.params[0]))
            for idx in range(1, len(inputs)):
                operation += (inputs[idx]*torch.sigmoid(self.params[idx]))
            return self.unit(operation)

        else:
            return self.unit(inputs[0])
        
        
class RandWireGraph(nn.Module):
    def __init__(self, num_nodes, graph_probability, in_channels, out_channels, train_mode, graph_name):
        super(RandWireGraph, self).__init__()
        
        self.num_nodes = num_nodes
        
        random_graph_node = RandomGraph(num_nodes, graph_probability)
        
        if train_mode:
            random_graph = random_graph_node.make_graph()
            self.nodes, self.incoming_edges = random_graph_node.get_graph_config(random_graph)
            random_graph_node.save_graph(random_graph, graph_name)

        else:
            random_graph = random_graph_node.load_graph(graph_name)
            self.nodes, self.incoming_edges = random_graph_node.get_graph_config(random_graph)

        self.module_lists = nn.ModuleList([GraphNode(self.incoming_edges[0], in_channels, out_channels, stride=2)])
        self.module_lists.extend([GraphNode(self.incoming_edges[n], out_channels, out_channels) for n in self.nodes if n > 0])

    def forward(self, x):
        cache = {}
        
        operation = self.module_lists[0].forward(x)
        cache[0] = operation
        
        for n in range(1, len(self.nodes) - 1):
            if len(self.incoming_edges[n]) > 1:
                operation = self.module_lists[n].forward(*[cache[incoming_vtx] for incoming_vtx in self.incoming_edges[n]])
            
            else:
                operation = self.module_lists[n].forward(cache[self.incoming_edges[n][0]])
            cache[n] = operation
            
        operation = cache[self.incoming_edges[self.num_nodes + 1][0]]
        for incomimg_vtx in range(1, len(self.incoming_edges[self.num_nodes + 1])):
            operation += cache[self.incoming_edges[self.num_nodes + 1][incomimg_vtx]]
        
        return operation / len(self.incoming_edges[self.num_nodes + 1])


class RandWireNet(nn.Module):
    def __init__(self, num_nodes, graph_probability, in_channels, out_channels, train_mode, dropout=0.3, class_num=10):
        super(RandWireNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        self.conv2 = nn.Sequential(
            RandWireGraph(num_nodes, graph_probability, out_channels, out_channels*2, train_mode, graph_name='conv2')
        )
        
        self.conv3 = nn.Sequential(
            RandWireGraph(num_nodes, graph_probability, out_channels*2, out_channels*4, train_mode, graph_name='conv3')
        )
        
        self.conv4 = nn.Sequential(
            RandWireGraph(num_nodes, graph_probability, out_channels*4, out_channels*8, train_mode, graph_name='conv4')
        )
        
        self.classifier = nn.Sequential(
            nn.Conv2d(out_channels*8, 1280, kernel_size=1),
            nn.BatchNorm2d(1280)
        )
        
        self.out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, class_num)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        
        _, _, h, w = x.shape
        x = nn.AvgPool2d(h, w)(x)
        x = x.squeeze()
        x = self.out(x)
        
        return x
    
    
if __name__ == '__main__':
    model = RandWireNet(32, 0.7, 3, 64, True).cuda()
    x = torch.randn(1, 3, 32, 32).cuda()
    assert model(x).shape == (10,)