!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    {%set comma, batchstr, fltr_axis = ":", "", 2 -%}
    subroutine forward_{{version}}{{dtype_token}}(x, y, {{batchstr}}csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: {{batchstr}}nnz, dim_in, dim_out, nfi,mode
        {{dtype}},intent(in) :: x({{batchstr}}nfi, dim_in)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        {{dtype}},intent(out) :: y({{batchstr}}nfi, dim_out)

        integer :: start_, end_, col
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, {{batchstr}}nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            if(mode==0) then
                y({{comma}},col)=maxval({%if is_complex%}abs{%endif%}(x({{comma}},csc_indices(start_:end_-1))),{{fltr_axis}})
            else
                y({{comma}},col)=sum(x({{comma}},csc_indices(start_:end_-1)),{{fltr_axis}})/(end_-start_)
            endif
        enddo
    end subroutine forward_{{version}}{{dtype_token}}

    subroutine backward_{{version}}{{dtype_token}}(dy,x,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,{{batchstr}}mode)
        implicit none
        integer,intent(in) :: {{batchstr}}nnz,dim_in,dim_out,nfi,mode
        {{dtype}},intent(in) :: x({{batchstr}}nfi, dim_in)
        {{dtype}},intent(in) :: dy({{batchstr}}nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        {{dtype}},intent(out) :: dx({{batchstr}}nfi, dim_in)

        integer :: start_, end_, col, irow, ib
        integer,pointer :: rows(:)
        {{dtype}} :: y_work(nfi)
        {{dtype}},parameter :: one={{dtype_one}}
        {{dtype}},parameter :: zero={{dtype_zero}}

        !f2py intent(in) x, dy, csc_indices, csc_indptr
        !f2py intent(in) nfi, {{batchstr}}dim_in, dim_out, nnz
        !f2py intent(out) dx

        if(mode==0) dx=0

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            if(mode==0) then
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc({%if is_complex%}abs{%endif%}(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            else
                y_work=dy({{comma}},col)/(end_-start_)
                do irow=start_,end_-1
                    dx({{comma}},csc_indices(irow))=y_work
                enddo
            endif
        enddo
    end subroutine backward_{{version}}{{dtype_token}}
    {%endfor -%}
end module lib
