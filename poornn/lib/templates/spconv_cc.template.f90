!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    ! return true if loc is in region, with canvas img_shape
    subroutine inregion(loc, region, img_shape, ndim, boundary, is_inregion)
        implicit none
        integer,intent(in) :: ndim, region(ndim), img_shape(ndim), boundary
        integer,intent(inout) :: loc(ndim)
        logical,intent(out) :: is_inregion
        if (boundary==1) loc = modulo(loc, img_shape)
        if (any(loc>=region)) then
            is_inregion = .false.
        else
            is_inregion = .true.
        endif
    end subroutine inregion

    {%for dtype in dtype_list -%}
    {%if dtype == "complex*16"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "dcmplx(1D0,0D0)", "dcmplx(0D0,0D0)", "z", True -%}
    {%elif dtype == "complex*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex ="cmplx(1.0,0.0)", "cmplx(0.0,0.0)", "c", True -%}
    {%elif dtype == "real*8"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1D0", "0D0", "d", False -%}
    {%elif dtype == "real*4"%}{%set dtype_one, dtype_zero, dtype_token, is_complex = "1.0", "0.0", "s", False -%}
    {%endif -%}
    {%for version in version_list -%}
    {%for withbatch in [True, False] -%}
    {%if withbatch%}{%set comma, num_batch, batch_token, batch_dim = ":,", "num_batch, ", "", "num_batch" -%}
    {%else%}{%set comma, num_batch, batch_token, batch_dim = "", "", "1", "1" -%}
    {%endif -%}
    subroutine forward{{batch_token}}_v1{{dtype_token}}(locs, dx, y, {{num_batch}}csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), {{num_batch}}nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        {{dtype}},intent(in) :: dx({{num_batch}}nfi, dim_in)
        {{dtype}},intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        {{dtype}},intent(inout) :: y({{num_batch}}nfo, dim_out)

        {{dtype}} :: x_work({{num_batch}}nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        {{dtype}},parameter :: one={{dtype_one}}
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, {{num_batch}}dim_out, nd, dim_in
        !f2py intent(inout) y

        locs_ = locs+1

        do col=1, dim_out
            start_ = csc_indptr(col)

            !prepair work space by taking rows in x.
            dxcount = 0
            do ii=1,nd
                ix = csc_indices(start_+ii-1)
                do idx=1,ndx
                    if (locs_(idx) == ix) then
                        dxcount = dxcount+1
                        x_work({{comma}}:,dxcount)=dx({{comma}}:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                {%if withbatch-%}
                call {{dtype_token}}gemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                    w_work, nfo, one, y(:,:,col), num_batch)
                {%else -%}
                call {{dtype_token}}gemv('N', nfo, k, one, w_work, nfo,&
                x_work, 1, one, y(:,col), 1)
                {%endif -%}
            endif
        enddo
    end subroutine forward{{batch_token}}_v1{{dtype_token}}

    subroutine forward{{batch_token}}_v2{{dtype_token}}(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, {{num_batch}}boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), {{num_batch}}dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        {{dtype}},intent(in) :: dx({{num_batch}}nfi, ndx)
        {{dtype}},intent(in) :: fltr_data(nfo, nfi, nd)
        {{dtype}},intent(inout) :: y({{num_batch}}nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        {{dtype}},parameter :: one={{dtype_one}}
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, {{num_batch}}dim_out, nd, kernel_shape
        !f2py intent(inout) y

        ! image shape cumprod, in fortran order ~
        img_shape_cumprod(1)=1
        do k=1,ndim-1
            img_shape_cumprod(k+1)=img_shape_cumprod(k)*img_shape(k)
        enddo

        do idx=1,ndx
            loci = modulo(locs(idx)/img_shape_cumprod, img_shape)
            do offseti=1,dim_out
                loci_ = loci - offset_table(:,offseti)
                call inregion(loci_, kernel_shape, img_shape, ndim, boundary, is_inregion)
                if (is_inregion) then
                    fltr_index = sum(loci_*img_shape_cumprod)
                    {%if withbatch-%}
                    call {{dtype_token}}gemm('N', 'T', num_batch, nfo, nfi, one, dx({{comma}}:,idx), num_batch,&
                        fltr_data(:,:,fltr_index+1), nfo, one, y({{comma}}:,offseti), num_batch)
                    {%else -%}
                    call {{dtype_token}}gemv('N', nfo, nfi, one, fltr_data(:,:,fltr_index+1), nfo,&
                    dx({{comma}}:,idx), 1, one, y({{comma}}:,offseti), 1)
                    {%endif -%}
                endif
            enddo
        enddo
    end subroutine forward{{batch_token}}_v2{{dtype_token}}

    {%endfor -%}
    {%endfor -%}
    {%endfor -%}
end module lib
