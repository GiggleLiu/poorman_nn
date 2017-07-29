#render templates
import os

template_list=['linear.template.f90', 'spconv.template.f90',\
        'pooling.template.f90','relu.template.f90']
source_list=['linear.f90', 'spconv.f90','pooling.f90', 'relu.f90']
extension_list=[source[:-4] for source in source_list]

libdir='poornn/lib'
#libdir='.'
def render_f90s():
    from frender import render_f90
    for template, source in zip(template_list, source_list):
        if not os.path.exists(os.path.join(libdir, source)):
            render_f90(libdir, os.path.join('templates', template),{
                'version_list':['general','contiguous'] if source=='spconv.f90' else [''],
                'dtype_list':['complex*16','real*8','real*4']
                }, out_file=os.path.join(libdir, source))
render_f90s()
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, NotFoundError, numpy_info
    config=Configuration('lib',parent_package,top_path)

    #get lapack options
    lapack_opt = get_info('lapack_opt')

    if not lapack_opt:
        raise NotFoundError('no lapack/blas resources found')

    atlas_version = ([v[3:-3] for k, v in lapack_opt.get('define_macros', [])
                      if k == 'ATLAS_INFO']+[None])[0]
    if atlas_version:
        print(('ATLAS version: %s' % atlas_version))

    #include_dirs=[os.curdir,'$MKLROOT/include']
    #library_dirs=['$MKLROOT/lib/intel64']
    #libraries=['mkl_intel_lp64','mkl_sequential','mkl_core', 'm', 'pthread']


    for extension, source in zip(extension_list, source_list):
        #config.add_extension(extension, [os.path.join(libdir, source)], libraries=libraries,
        #        library_dirs=library_dirs, include_dirs=include_dirs)
        config.add_extension(extension, [os.path.join(libdir, source)], extra_info=lapack_opt)
    return config
