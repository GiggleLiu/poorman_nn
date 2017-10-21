'''
Set up file for poornn.
'''


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
    config = Configuration('poornn', parent_package, top_path)
    config.add_subpackage('lib')
    return config
