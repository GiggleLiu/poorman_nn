import pdb
from jinja2 import Template
from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
        loader=FileSystemLoader('templates'),
        )

def render_f90(template_file,var_dict,out_file):
    res=env.get_template(template_file).render(var_dict)
    with open(out_file,'w') as of:
        of.write('!This is an f90 file automatically generated.\n'+res)

if __name__=='__main__':
    render_f90('spconv.template.f90',{'version_list':['general','contiguous'],'dtype_list':['complex*16','real*8','real*4']},out_file='spconv.f90')
    render_f90('linear.template.f90',{'version_list':[''],'dtype_list':['complex*16','real*8','real*4']},out_file='linear.f90')
