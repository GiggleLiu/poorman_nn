import django,pdb
from django.conf import settings
from django.template import Template, Context
from django.template.loader import get_template

settings.configure(TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': ['.'],
        'APP_DIRS': True,
        'OPTIONS': {
            # ... some options here ...
        },
    },
])

django.setup()

var_complex={'dtype':'complex*16',
    'dtype_token':'z',
    'dtype_one':'dcmplx(1D0,0D0)',
    'dtype_zero':'dcmplx(0D0,0D0)',
    'is_complex':True
    }
var_real={'dtype':'real*4',
    'dtype_token':'s',
    'dtype_one':'1.0',
    'dtype_zero':'0.0',
    'is_complex':False,
    }

def reder_f90(template_file,var_dict,out_file):
    res=get_template(template_file).render(var_dict)
    with open(out_file,'w') as of:
        of.write(res)

if __name__=='__main__':
    var_dict={'version_list':['general','contiguous']}
    var_dict.update(var_complex)
    reder_f90('spconv.template.f90',
            var_dict=var_dict,
            out_file='spconvz.f90')
    var_dict.update(var_real)
    reder_f90('spconv.template.f90',
            var_dict=var_dict,
            out_file='spconvd.f90')
