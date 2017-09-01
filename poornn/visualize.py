'''
Visualization for neural networks.
'''
from utils import _connect

__all__ = ['viznn']

def viznn(nn, filename=None):
    '''
    Visualize a neural network.

    Parameters:
        :nn: <NN>,
        :filename: str, to filename to save, default is "G.sv"
    '''
    from pygraphviz import AGraph
    g=AGraph(directed=True, rankdir='TD', compound=True)
    g.add_node('x')
    father = nn.__graphviz__(g, father=g.get_node('x'))
    g.add_node('y')
    _connect(g, father, g.get_node('y'), nn.output_shape, nn.otype, pos='last')
    g.layout('dot')
    g.draw(filename)
