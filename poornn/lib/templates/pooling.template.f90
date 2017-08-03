!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    !mode = 0: max real.
    !mode = 1: max abs.
    !mode = 2: min real.
    !mode = 3: min abs.
    !mode = 4: mean pooling.
    subroutine forward_{{version}}{{dtype_token}}(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        {{dtype}},intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        {{dtype}},intent(out) :: y(nfi, dim_out)

        integer,pointer :: rows(:)
        integer :: start_, end_, col, ib, irow
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            select case (mode)
            case (0)
                {%if is_complex-%}
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(real(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
                {%else-%}
                y(:,col)=maxval(x(:,csc_indices(start_:end_-1)),2)
                {%endif-%}
            case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (2)
                {%if is_complex-%}
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(real(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
                {%else-%}
                y(:,col)=minval(x(:,csc_indices(start_:end_-1)),2)
                {%endif-%}
            case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (4)
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
        enddo
    end subroutine forward_{{version}}{{dtype_token}}

    subroutine backward_{{version}}{{dtype_token}}(dy,x,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,mode)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,mode
        {{dtype}},intent(in) :: x(nfi, dim_in)
        {{dtype}},intent(in) :: dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        {{dtype}},intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, irow, ib
        integer,pointer :: rows(:)
        {{dtype}} :: y_work(nfi)
        {{dtype}},parameter :: one={{dtype_one}}
        {{dtype}},parameter :: zero={{dtype_zero}}

        !f2py intent(in) x, dy, csc_indices, csc_indptr
        !f2py intent(in) nfi, dim_in, dim_out, nnz
        !f2py intent(out) dx

        if(mode==0) dx=0

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            select case (mode)
            case (0)
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc({%if is_complex%}real{%endif%}(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (2)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc({%if is_complex%}real{%endif%}(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (4)
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
        enddo
    end subroutine backward_{{version}}{{dtype_token}}
    {%endfor -%}
end module lib
