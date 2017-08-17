module lib
    character,parameter :: matdescra(6)=(/'G','-','-','F','-','-'/)
    contains
    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    subroutine forward{{dtype_token}}(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        {{dtype}},intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        {{dtype}},intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)

        integer :: k

        do k=1,num_batch
            y(k,:)=bias
            call mkl_{{dtype_token}}cscmv('T', dim_in, dim_out, {{dtype_one}}, matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), x(k,:), {{dtype_one}}, y(k,:))
        enddo
    end subroutine forward{{dtype_token}}

    {%for version in version_list -%}
    {%if version == "_conv"%}{%set nd_, nd = "nd, ", "nd" -%}
    {%else%}{%set nd_, nd = "", "nnz"-%}
    {%endif -%}
    subroutine backward{{version}}{{dtype_token}}(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            {{data_indices}}csc_data, nnz,{{nd_}}dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,{{nd_}}dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        {{dtype}},intent(out) :: dx(num_batch, dim_in)
        {{dtype}},intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data({{nd}})
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        {{dtype}},intent(out) :: dweight({{nd}}), dbias(dim_out)

        integer :: k, col, start_, end_{%if version == '_conv'%},data_start_, data_end_, ifo{%endif%}

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    {%if version == "_conv" -%}
                    data_start_=modulo(start_-1, nd)+1
                    data_end_=modulo(end_-1, nd)+1
                    dweight(data_start_:data_end_)= dweight(data_start_:data_end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    {%else-%}
                    dweight(start_:end_)= dweight(start_:end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    {%endif-%}
                enddo
            endif
            if(do_xgrad) then
                !calculate dx
                {%if version == '_conv' -%}
                do ifo=1,nnz/nd
                    call mkl_{{dtype_token}}cscmv('N', dim_in, dim_out, {{dtype_one}}, matdescra, csc_data, csc_indices,&
                    csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), {{dtype_zero}}, dx(k,:))
                enddo
                {%endif%}
                call mkl_{{dtype_token}}cscmv('N', dim_in, dim_out, {{dtype_one}}, matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), {{dtype_zero}}, dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward{{version}}{{dtype_token}}
    {%endfor -%}
    {%endfor -%}
end module lib
