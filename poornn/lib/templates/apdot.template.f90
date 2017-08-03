!orders: batch_dim, feature_dim_out/in
module lib
    contains
    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    !y = y + A^x, with ^ the apdot operation mark.
    subroutine apdot{{dtype_token}}(mat, x, y, m, n)
        implicit none
        integer,intent(in) :: m, n
        {{dtype}},intent(in) :: mat(m,n), x(n)
        {{dtype}},intent(inout) :: y(m)
        do j=1,n
            do i=1,m
                y(i)=y(i)*(mat(i,j)+x(j))
            enddo
        enddo
    end subroutine apdot{{dtype_token}}

    {%for version in version_list -%}
    subroutine forward_{{version}}{{dtype_token}}(x, y, weight,{%if version == "masked"%} mask,{%endif%} bias, num_batch, nfi, nfo)
        implicit none
        integer,intent(in) :: num_batch, nfi, nfo
        {{dtype}},intent(in) :: x(num_batch, nfi), weight(nfo, nfi), bias(nfo)
        {%if version == "masked"%}logical,intent(in) :: mask(nfo, nfi){%endif%}
        {{dtype}},intent(out) :: y(num_batch, nfo)
        {{dtype}},parameter :: one={{dtype_one}}
        integer :: i

        do i=1,num_batch
            y(i)=product(weight+x(i),2)*bias(i)
        enddo
    end subroutine forward_{{version}}{{dtype_token}}

    subroutine backward_{{version}}{{dtype_token}}(dy,x, weight, dx, dweight,dbias{%if version == "masked"%},mask{%endif%},&
            nfi,nfo, num_batch, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch,nfi,nfo
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        {{dtype}},intent(in) :: x(num_batch, nfi), dy(num_batch, nfo), weight(nfo, nfi)
        {%if version == "masked"%}logical,intent(in) :: mask(nfo, nfi){%endif%}
        {{dtype}},intent(out) :: dweight(nfo, nfi), dbias(nfo), dx(num_batch, nfi)

        {{dtype}},parameter :: one={{dtype_one}}
        {{dtype}},parameter :: zero={{dtype_zero}}

        !f2py intent(out) dx, dweight, dbias

        if(do_wgrad) then
            !call {{dtype_token}}gemm('T', 'N', nfo, nfi, num_batch, one, dy, num_batch,&
            !    {%if is_complex%}conjg(x){%else%}x{%endif%}, num_batch, zero, dweight, nfo)
            call {{dtype_token}}gemm({%if is_complex%}'C'{%else%}'T'{%endif%}, 'N', nfo, nfi, num_batch, one, dy, num_batch,&
                x, num_batch, zero, dweight, nfo)
        endif
        if(do_xgrad) then
            !call {{dtype_token}}gemm('N', 'N', num_batch, nfi, nfo, one, dy, num_batch,&
            !    {%if is_complex%}conjg(weight){%else%}weight{%endif%}, nfo, zero, dx, num_batch)
            call {{dtype_token}}gemm('N', 'N', num_batch, nfi, nfo, one, {%if is_complex%}conjg(dy){%else%}dy{%endif%}, num_batch,&
                weight, nfo, zero, dx, num_batch)
        endif
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_{{version}}{{dtype_token}}
    {%endfor -%}
    {%endfor -%}
end module lib
