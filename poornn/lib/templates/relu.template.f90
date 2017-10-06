!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
!version: 1 -> real-imagine seperate, otherwise real only
module lib
    contains
    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set rtype, icmplx, dtype_token, is_complex = "real*8", "dcmplx", "z", True -%}
    {%elif dtype == "complex*8"%}{%set rtype, icmplx, dtype_token, is_complex ="real*4", "cmplx", "c", True -%}
    {%elif dtype == "real*8"%}{%set rtype, dtype_token, is_complex = "real*8", "d", False -%}
    {%elif dtype == "real*4"%}{%set rtype, dtype_token, is_complex = "real*4", "s", False -%}
    {%endif-%}
    {%if is_complex%}{%set version_list = ["r","ri"]-%}{%else%}{%set version_list = ["r"]-%}{%endif-%}
    {%for version in version_list-%}
    subroutine forward_{{version}}{{dtype_token}}(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        {{rtype}},intent(in) :: leak
        {{dtype}},intent(in) :: x(dim_in)
        {{dtype}},intent(out) :: y(dim_in)
        integer :: i
        {{dtype}} :: xi
        do i=1,dim_in
            xi=x(i)
            {%if version == "r"-%}
            if({%if is_complex%}real(xi)<0{%else%}xi<0{%endif%}) then
                y(i)=leak*xi
            else
                y(i)=xi
            endif
            {%else-%}
            if(aimag(xi)>0 .and. real(xi)>0) then
                y(i)=xi
            else if(aimag(xi)>0 .and. real(xi)<0) then
                y(i)={{icmplx}}(leak*real(xi),aimag(xi))
            else if(real(xi)<0) then
                y(i)=leak*xi
            else
                y(i)={{icmplx}}(real(xi),leak*aimag(xi))
            endif
            {%-endif%}
        enddo
    end subroutine forward_{{version}}{{dtype_token}}

    subroutine backward_{{version}}{{dtype_token}}(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        {{rtype}},intent(in) :: leak
        {{dtype}},intent(in) :: x(dim_in)
        {{dtype}},intent(in) :: dy(dim_in)
        {{dtype}},intent(out) :: dx(dim_in)
        {{dtype}} :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            {%if version == "r"-%}
            if({%if is_complex%}real(xi)<0{%else%}xi<0{%endif%}) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dy(i)
            endif
            {%else-%}
            if(aimag(xi)>0 .and. real(xi)>0) then
                dx(i)=dy(i)
            else if(aimag(xi)>0 .and. real(xi)<0) then
                dx(i)={{icmplx}}(leak*real(dy(i)),aimag(dy(i)))
            else if(real(xi)<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)={{icmplx}}(real(dy(i)),leak*aimag(dy(i)))
            endif
            {%-endif%}
        enddo
    end subroutine backward_{{version}}{{dtype_token}}
    {%endfor%}
    {%endfor%}
end module lib
