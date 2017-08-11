!This is an f90 file automatically generated.
!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    character,parameter :: matdescra(6)=(/'G','-','-','F','-','-'/)
    contains
    subroutine forwardz(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        complex*16,intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        complex*16,intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_in+1)
        integer :: k
        !f2py intent(in) x, csc_indices, csc_indptr, csc_data, dim_in, nnz, dim_out, num_batch
        !f2py intent(out) y
        do k=1,num_batch
            y(k,:)=bias
            call mkl_zcscmv('N', dim_out, dim_in, dcmplx(1D0,0D0), matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), x(k,:), dcmplx(1D0,0D0), y(k,:))
        enddo
    end subroutine forwardz

    subroutine backwardz(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(out) :: dx(num_batch, dim_in)
        complex*16,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nnz)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_in+1)
        complex*16,intent(out) :: dweight(nnz), dbias(dim_out)

        integer :: k, col, start_, end_

        !f2py intent(in) x, dy, csc_indices, csc_indptr, csc_data, dim_in, dim_out, nnz, do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) num_batch
        !f2py intent(out) dx, dweight, dbias

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_in
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    dweight(start_:end_)= dweight(start_:end_)+x(k,col)*dy(k,csc_indices(start_:end_))
                enddo
            endif
            if(do_xgrad) then
                !calculate dx
                call mkl_zcscmv('T', dim_out, dim_in, dcmplx(1D0,0D0), matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), dcmplx(0D0,0D0), dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backwardz
    subroutine forwardd(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        real*8,intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        real*8,intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_in+1)
        integer :: k
        !f2py intent(in) x, csc_indices, csc_indptr, csc_data, dim_in, nnz, dim_out, num_batch
        !f2py intent(out) y
        do k=1,num_batch
            y(k,:)=bias
            call mkl_dcscmv('N', dim_out, dim_in, 1D0, matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), x(k,:), 1D0, y(k,:))
        enddo
    end subroutine forwardd

    subroutine backwardd(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(out) :: dx(num_batch, dim_in)
        real*8,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nnz)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_in+1)
        real*8,intent(out) :: dweight(nnz), dbias(dim_out)

        integer :: k, col, start_, end_

        !f2py intent(in) x, dy, csc_indices, csc_indptr, csc_data, dim_in, dim_out, nnz, do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) num_batch
        !f2py intent(out) dx, dweight, dbias

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_in
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    dweight(start_:end_)= dweight(start_:end_)+x(k,col)*dy(k,csc_indices(start_:end_))
                enddo
            endif
            if(do_xgrad) then
                !calculate dx
                call mkl_dcscmv('T', dim_out, dim_in, 1D0, matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), 0D0, dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backwardd
    subroutine forwards(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        real*4,intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        real*4,intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_in+1)
        integer :: k
        !f2py intent(in) x, csc_indices, csc_indptr, csc_data, dim_in, nnz, dim_out, num_batch
        !f2py intent(out) y
        do k=1,num_batch
            y(k,:)=bias
            call mkl_scscmv('N', dim_out, dim_in, 1.0, matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), x(k,:), 1.0, y(k,:))
        enddo
    end subroutine forwards

    subroutine backwards(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(out) :: dx(num_batch, dim_in)
        real*4,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nnz)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_in+1)
        real*4,intent(out) :: dweight(nnz), dbias(dim_out)

        integer :: k, col, start_, end_

        !f2py intent(in) x, dy, csc_indices, csc_indptr, csc_data, dim_in, dim_out, nnz, do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) num_batch
        !f2py intent(out) dx, dweight, dbias

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_in
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    dweight(start_:end_)= dweight(start_:end_)+x(k,col)*dy(k,csc_indices(start_:end_))
                enddo
            endif
            if(do_xgrad) then
                !calculate dx
                call mkl_scscmv('T', dim_out, dim_in, 1.0, matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), 0.0, dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backwards
    end module lib