{%for dtype in dtype_list -%}
{%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
{%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
{%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
{%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
{%endif -%}
subroutine fsign_{{dtype_token}}(x, nx, s)
    implicit none
    integer,intent(in) :: nx
    {{dtype}},intent(in) :: x(nx)
    {{dtype}},intent(out) :: s(nx)

    {{dtype}} :: xi
    integer :: i

    do i=1,nx
        xi = x(i)
        if(xi==0) then
            s(i) = 0
        else
            s(i) = xi/abs(xi)
        endif
    enddo
end subroutine fsign_{{dtype_token}}
{%endfor-%}
