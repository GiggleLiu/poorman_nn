!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    subroutine forward_{{version}}{{dtype_token}}(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        {{dtype}},intent(in) :: leak
        {{dtype}},intent(in) :: x(dim_in)
        {{dtype}},intent(out) :: y(dim_in)
        integer :: i
        {{dtype}} :: xi
        do i=1,dim_in
            xi=x(i)
            if({%if is_complex%}aimag(xi)<0 .or. real(xi)<0{%else%}xi<0{%endif%}) then
            !if({%if is_complex%}real(xi)<0{%else%}xi<0{%endif%}) then
                y(i)=leak*xi
            else
                y(i)=xi
            endif
        enddo
    end subroutine forward_{{version}}{{dtype_token}}

    subroutine backward_{{version}}{{dtype_token}}(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        {{dtype}},intent(in) :: leak
        {{dtype}},intent(in) :: x(dim_in)
        {{dtype}},intent(in) :: dy(dim_in)
        {{dtype}},intent(out) :: dx(dim_in)
        {{dtype}} :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if({%if is_complex%}aimag(xi)<0 .or. real(xi)<0{%else%}xi<0{%endif%}) then
            !if({%if is_complex%}real(xi)<0{%else%}xi<0{%endif%}) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dy(i)
            endif
        enddo
    end subroutine backward_{{version}}{{dtype_token}}
    {%endfor%}
end module lib
