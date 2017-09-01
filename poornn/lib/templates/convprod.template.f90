module lib
    contains
    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    subroutine forward_{{version}}{{dtype_token}}(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, powers, nd)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nd
        {{dtype}},intent(in) :: x(nfi, dim_in)
        {{dtype}},intent(in):: powers(nd)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        {{dtype}},intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col, irow

        y=1
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            do irow=1,end_-start_
                y(:,col)=y(:,col)*x(:,csc_indices(start_+irow-1))**powers(irow)
            enddo
        enddo
    end subroutine forward_{{version}}{{dtype_token}}

    subroutine backward_{{version}}{{dtype_token}}(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,powers,nd)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi, nd
        {{dtype}},intent(in) :: x(nfi, dim_in)
        {{dtype}},intent(in):: powers(nd)
        {{dtype}},intent(in) :: y(nfi, dim_out), dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        {{dtype}},intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, ib
        integer,pointer :: rows(:)

        dx=0
        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            rows=>csc_indices(start_:end_-1)
            do ib=1,nfi
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)*powers/x(ib,rows)
            enddo
        enddo
    end subroutine backward_{{version}}{{dtype_token}}
    {%endfor -%}
end module lib
