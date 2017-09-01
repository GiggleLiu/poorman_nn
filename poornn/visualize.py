'''
Visualization for neural networks.
'''

__all__ = ['viznn']

def viznn(nn, filename=None):
    '''
    Visualize a neural network.

    Parameters:
        :nn: <NN>,
        :filename: str, to filename to save, default is "G.sv"
    '''
    from graphviz import Digraph
    g=Digraph('G',filename=filename)
    g.node('x')
    father = nn.__graphviz__(g, father='x')
    g.node('y')
    g.edge(father, 'y', label='<<font point-size="10px">%s</font><br align="left"/>\
<font point-size="10px">%s</font><br align="left"/>>'%(nn.output_shape, nn.otype))
    if filename is None:
        g.view()
    else:
        g.render(view=False)
