#render templates
import os

template_list=['linear.template.f90', 'spconv.template.f90']
source_list=['linear.f90', 'spconv.f90']
extension_list=[source[:-4] for source in source_list]

libdir='poornn/lib'
#libdir='.'
def render_f90s():
    from frender import render_f90
    for template, source in zip(template_list, source_list):
        if not os.path.exists(os.path.join(libdir, source)):
            render_f90(libdir, os.path.join(libdir, 'templates', template),{
                'version_list':['general','contiguous'],
                'dtype_list':['complex*16','real*8','real*4']
                }, out_file=os.path.join(libdir, source))
render_f90s()
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    os.environ["CC"] = "gfortran"
    os.environ["CXX"] = "gfortran"
    include_dirs=[os.curdir,'/intel/mkl/include']
    library_dirs=[os.path.expanduser('~')+'/intel/mkl/lib/intel64']
    libraries=['mkl_intel_lp64','mkl_sequential','mkl_core', 'm', 'pthread']

    config=Configuration('lib',parent_package,top_path)
    for extension, source in zip(extension_list, source_list):
        config.add_extension(extension, [os.path.join(libdir, source)], libraries=libraries,
                library_dirs=library_dirs, include_dirs=include_dirs)
    return config
