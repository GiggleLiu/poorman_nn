!This is an f90 file automatically generated.
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

    subroutine forward_v1z(locs, dx, y, num_batch, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        complex*16,intent(in) :: dx(num_batch, nfi, dim_in)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(inout) :: y(num_batch, nfo, dim_out)

        complex*16 :: x_work(num_batch, nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, num_batch, dim_out, nd, dim_in
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
                        x_work(:,:,dxcount)=dx(:,:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call zgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                    w_work, nfo, one, y(:,:,col), num_batch)
                endif
        enddo
    end subroutine forward_v1z

    subroutine forward_v2z(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, num_batch, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        complex*16,intent(in) :: dx(num_batch, nfi, ndx)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        complex*16,intent(inout) :: y(num_batch, nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, num_batch, dim_out, nd, kernel_shape
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
                    call zgemm('N', 'T', num_batch, nfo, nfi, one, dx(:,:,idx), num_batch,&
                        fltr_data(:,:,fltr_index+1), nfo, one, y(:,:,offseti), num_batch)
                    endif
            enddo
        enddo
    end subroutine forward_v2z

    subroutine forward1_v1z(locs, dx, y, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        complex*16,intent(in) :: dx(nfi, dim_in)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(inout) :: y(nfo, dim_out)

        complex*16 :: x_work(nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, dim_out, nd, dim_in
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
                        x_work(:,dxcount)=dx(:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call zgemv('N', nfo, k, one, w_work, nfo,&
                x_work, 1, one, y(:,col), 1)
                endif
        enddo
    end subroutine forward1_v1z

    subroutine forward1_v2z(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        complex*16,intent(in) :: dx(nfi, ndx)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        complex*16,intent(inout) :: y(nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, dim_out, nd, kernel_shape
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
                    call zgemv('N', nfo, nfi, one, fltr_data(:,:,fltr_index+1), nfo,&
                    dx(:,idx), 1, one, y(:,offseti), 1)
                    endif
            enddo
        enddo
    end subroutine forward1_v2z

    subroutine forward_v1c(locs, dx, y, num_batch, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        complex*8,intent(in) :: dx(num_batch, nfi, dim_in)
        complex*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*8,intent(inout) :: y(num_batch, nfo, dim_out)

        complex*8 :: x_work(num_batch, nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        complex*8,parameter :: one=cmplx(1.0,0.0)
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, num_batch, dim_out, nd, dim_in
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
                        x_work(:,:,dxcount)=dx(:,:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call cgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                    w_work, nfo, one, y(:,:,col), num_batch)
                endif
        enddo
    end subroutine forward_v1c

    subroutine forward_v2c(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, num_batch, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        complex*8,intent(in) :: dx(num_batch, nfi, ndx)
        complex*8,intent(in) :: fltr_data(nfo, nfi, nd)
        complex*8,intent(inout) :: y(num_batch, nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        complex*8,parameter :: one=cmplx(1.0,0.0)
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, num_batch, dim_out, nd, kernel_shape
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
                    call cgemm('N', 'T', num_batch, nfo, nfi, one, dx(:,:,idx), num_batch,&
                        fltr_data(:,:,fltr_index+1), nfo, one, y(:,:,offseti), num_batch)
                    endif
            enddo
        enddo
    end subroutine forward_v2c

    subroutine forward1_v1c(locs, dx, y, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        complex*8,intent(in) :: dx(nfi, dim_in)
        complex*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*8,intent(inout) :: y(nfo, dim_out)

        complex*8 :: x_work(nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        complex*8,parameter :: one=cmplx(1.0,0.0)
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, dim_out, nd, dim_in
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
                        x_work(:,dxcount)=dx(:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call cgemv('N', nfo, k, one, w_work, nfo,&
                x_work, 1, one, y(:,col), 1)
                endif
        enddo
    end subroutine forward1_v1c

    subroutine forward1_v2c(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        complex*8,intent(in) :: dx(nfi, ndx)
        complex*8,intent(in) :: fltr_data(nfo, nfi, nd)
        complex*8,intent(inout) :: y(nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        complex*8,parameter :: one=cmplx(1.0,0.0)
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, dim_out, nd, kernel_shape
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
                    call cgemv('N', nfo, nfi, one, fltr_data(:,:,fltr_index+1), nfo,&
                    dx(:,idx), 1, one, y(:,offseti), 1)
                    endif
            enddo
        enddo
    end subroutine forward1_v2c

    subroutine forward_v1d(locs, dx, y, num_batch, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        real*8,intent(in) :: dx(num_batch, nfi, dim_in)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(inout) :: y(num_batch, nfo, dim_out)

        real*8 :: x_work(num_batch, nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        real*8,parameter :: one=1D0
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, num_batch, dim_out, nd, dim_in
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
                        x_work(:,:,dxcount)=dx(:,:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call dgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                    w_work, nfo, one, y(:,:,col), num_batch)
                endif
        enddo
    end subroutine forward_v1d

    subroutine forward_v2d(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, num_batch, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        real*8,intent(in) :: dx(num_batch, nfi, ndx)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        real*8,intent(inout) :: y(num_batch, nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        real*8,parameter :: one=1D0
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, num_batch, dim_out, nd, kernel_shape
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
                    call dgemm('N', 'T', num_batch, nfo, nfi, one, dx(:,:,idx), num_batch,&
                        fltr_data(:,:,fltr_index+1), nfo, one, y(:,:,offseti), num_batch)
                    endif
            enddo
        enddo
    end subroutine forward_v2d

    subroutine forward1_v1d(locs, dx, y, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        real*8,intent(in) :: dx(nfi, dim_in)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(inout) :: y(nfo, dim_out)

        real*8 :: x_work(nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        real*8,parameter :: one=1D0
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, dim_out, nd, dim_in
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
                        x_work(:,dxcount)=dx(:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call dgemv('N', nfo, k, one, w_work, nfo,&
                x_work, 1, one, y(:,col), 1)
                endif
        enddo
    end subroutine forward1_v1d

    subroutine forward1_v2d(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        real*8,intent(in) :: dx(nfi, ndx)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        real*8,intent(inout) :: y(nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        real*8,parameter :: one=1D0
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, dim_out, nd, kernel_shape
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
                    call dgemv('N', nfo, nfi, one, fltr_data(:,:,fltr_index+1), nfo,&
                    dx(:,idx), 1, one, y(:,offseti), 1)
                    endif
            enddo
        enddo
    end subroutine forward1_v2d

    subroutine forward_v1s(locs, dx, y, num_batch, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        real*4,intent(in) :: dx(num_batch, nfi, dim_in)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(inout) :: y(num_batch, nfo, dim_out)

        real*4 :: x_work(num_batch, nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        real*4,parameter :: one=1.0
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, num_batch, dim_out, nd, dim_in
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
                        x_work(:,:,dxcount)=dx(:,:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call sgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                    w_work, nfo, one, y(:,:,col), num_batch)
                endif
        enddo
    end subroutine forward_v1s

    subroutine forward_v2s(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, num_batch, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), num_batch, dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        real*4,intent(in) :: dx(num_batch, nfi, ndx)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        real*4,intent(inout) :: y(num_batch, nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        real*4,parameter :: one=1.0
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, num_batch, dim_out, nd, kernel_shape
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
                    call sgemm('N', 'T', num_batch, nfo, nfi, one, dx(:,:,idx), num_batch,&
                        fltr_data(:,:,fltr_index+1), nfo, one, y(:,:,offseti), num_batch)
                    endif
            enddo
        enddo
    end subroutine forward_v2s

    subroutine forward1_v1s(locs, dx, y, csc_indptr, csc_indices, fltr_data, &
            nnz, dim_in, dim_out, nfi, nfo, nd, ndx)
        implicit none
        integer,intent(in) :: locs(ndx), nnz, dim_in, dim_out, nfi, nfo, nd, ndx
        real*4,intent(in) :: dx(nfi, dim_in)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(inout) :: y(nfo, dim_out)

        real*4 :: x_work(nfi, ndx), w_work(nfo, nfi, ndx)
        integer :: start_, col, ii, k, ix, dxcount, idx, locs_(ndx)
        real*4,parameter :: one=1.0
        !f2py intent(in) locs, dx, csc_indices, csc_indptr, fltr_data,
        !f2py intent(in) nfi, nfo, ndx, nnz, dim_out, nd, dim_in
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
                        x_work(:,dxcount)=dx(:,idx)
                        w_work(:,:,dxcount)=fltr_data(:,:,ii)
                    endif
                enddo
            enddo
            if (dxcount/=0) then
                k=nfi*dxcount
                call sgemv('N', nfo, k, one, w_work, nfo,&
                x_work, 1, one, y(:,col), 1)
                endif
        enddo
    end subroutine forward1_v1s

    subroutine forward1_v2s(locs, dx, y, fltr_data, offset_table, img_shape,&
            dim_out, nfi, nfo, nd, ndx, boundary, kernel_shape, ndim)
        implicit none
        integer,intent(in) :: locs(ndx), dim_out, nfi, nfo, nd
        integer,intent(in) :: ndx, ndim, img_shape(ndim), boundary, offset_table(ndim, dim_out), kernel_shape(ndim)
        real*4,intent(in) :: dx(nfi, ndx)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        real*4,intent(inout) :: y(nfo, dim_out)

        integer :: k, idx, img_shape_cumprod(ndim), fltr_index, offseti, loci_(ndim), loci(ndim)
        real*4,parameter :: one=1.0
        logical :: is_inregion
        !f2py intent(in) locs, dx, fltr_data, ndim, offset_table, img_shape
        !f2py intent(in) nfi, nfo, ndx, dim_out, nd, kernel_shape
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
                    call sgemv('N', nfo, nfi, one, fltr_data(:,:,fltr_index+1), nfo,&
                    dx(:,idx), 1, one, y(:,offseti), 1)
                    endif
            enddo
        enddo
    end subroutine forward1_v2s

    end module lib