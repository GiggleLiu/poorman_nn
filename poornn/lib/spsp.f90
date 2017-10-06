!This is an f90 file automatically generated.
module lib
    character,parameter :: matdescra(6)=(/'G','-','-','F','-','-'/)
    contains
    subroutine forwardz(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        complex*16,intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        complex*16,intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)

        integer :: k

        do k=1,num_batch
            y(k,:)=bias
            call mkl_zcscmv('T', dim_in, dim_out, dcmplx(1D0,0D0), matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), x(k,:), dcmplx(1D0,0D0), y(k,:))
        enddo
    end subroutine forwardz

    subroutine backwardz(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(out) :: dx(num_batch, dim_in)
        complex*16,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nnz)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(out) :: dweight(nnz), dbias(dim_out)

        integer :: k, col, start_, end_

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    dweight(start_:end_)= dweight(start_:end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                
                call mkl_zcscmv('N', dim_in, dim_out, dcmplx(1D0,0D0), matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), dcmplx(0D0,0D0), dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backwardz
    subroutine backward_convz(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,nd, dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,nd, dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(out) :: dx(num_batch, dim_in)
        complex*16,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(out) :: dweight(nd), dbias(dim_out)

        integer :: k, col, start_, end_,data_start_, data_end_, ifo

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    data_start_=modulo(start_-1, nd)+1
                    data_end_=modulo(end_-1, nd)+1
                    dweight(data_start_:data_end_)= dweight(data_start_:data_end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                do ifo=1,nnz/nd
                    call mkl_zcscmv('N', dim_in, dim_out, dcmplx(1D0,0D0), matdescra, csc_data, csc_indices,&
                    csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), dcmplx(0D0,0D0), dx(k,:))
                enddo
                
                call mkl_zcscmv('N', dim_in, dim_out, dcmplx(1D0,0D0), matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), dcmplx(0D0,0D0), dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_convz
    subroutine forwardc(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        complex*8,intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        complex*8,intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)

        integer :: k

        do k=1,num_batch
            y(k,:)=bias
            call mkl_ccscmv('T', dim_in, dim_out, cmplx(1.0,0.0), matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), x(k,:), cmplx(1.0,0.0), y(k,:))
        enddo
    end subroutine forwardc

    subroutine backwardc(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*8,intent(out) :: dx(num_batch, dim_in)
        complex*8,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nnz)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*8,intent(out) :: dweight(nnz), dbias(dim_out)

        integer :: k, col, start_, end_

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    dweight(start_:end_)= dweight(start_:end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                
                call mkl_ccscmv('N', dim_in, dim_out, cmplx(1.0,0.0), matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), cmplx(0.0,0.0), dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backwardc
    subroutine backward_convc(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,nd, dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,nd, dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*8,intent(out) :: dx(num_batch, dim_in)
        complex*8,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*8,intent(out) :: dweight(nd), dbias(dim_out)

        integer :: k, col, start_, end_,data_start_, data_end_, ifo

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    data_start_=modulo(start_-1, nd)+1
                    data_end_=modulo(end_-1, nd)+1
                    dweight(data_start_:data_end_)= dweight(data_start_:data_end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                do ifo=1,nnz/nd
                    call mkl_ccscmv('N', dim_in, dim_out, cmplx(1.0,0.0), matdescra, csc_data, csc_indices,&
                    csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), cmplx(0.0,0.0), dx(k,:))
                enddo
                
                call mkl_ccscmv('N', dim_in, dim_out, cmplx(1.0,0.0), matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), cmplx(0.0,0.0), dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_convc
    subroutine forwardd(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        real*8,intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        real*8,intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)

        integer :: k

        do k=1,num_batch
            y(k,:)=bias
            call mkl_dcscmv('T', dim_in, dim_out, 1D0, matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), x(k,:), 1D0, y(k,:))
        enddo
    end subroutine forwardd

    subroutine backwardd(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(out) :: dx(num_batch, dim_in)
        real*8,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nnz)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(out) :: dweight(nnz), dbias(dim_out)

        integer :: k, col, start_, end_

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    dweight(start_:end_)= dweight(start_:end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                
                call mkl_dcscmv('N', dim_in, dim_out, 1D0, matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), 0D0, dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backwardd
    subroutine backward_convd(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,nd, dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,nd, dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(out) :: dx(num_batch, dim_in)
        real*8,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(out) :: dweight(nd), dbias(dim_out)

        integer :: k, col, start_, end_,data_start_, data_end_, ifo

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    data_start_=modulo(start_-1, nd)+1
                    data_end_=modulo(end_-1, nd)+1
                    dweight(data_start_:data_end_)= dweight(data_start_:data_end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                do ifo=1,nnz/nd
                    call mkl_dcscmv('N', dim_in, dim_out, 1D0, matdescra, csc_data, csc_indices,&
                    csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), 0D0, dx(k,:))
                enddo
                
                call mkl_dcscmv('N', dim_in, dim_out, 1D0, matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), 0D0, dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_convd
    subroutine forwards(x, y, bias, num_batch, csc_indptr, csc_indices, csc_data, nnz, dim_in, dim_out)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out
        real*4,intent(in) :: x(num_batch, dim_in), csc_data(nnz), bias(dim_out)
        real*4,intent(out),dimension(num_batch, dim_out) :: y
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)

        integer :: k

        do k=1,num_batch
            y(k,:)=bias
            call mkl_scscmv('T', dim_in, dim_out, 1.0, matdescra, csc_data, csc_indices,&
            csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), x(k,:), 1.0, y(k,:))
        enddo
    end subroutine forwards

    subroutine backwards(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(out) :: dx(num_batch, dim_in)
        real*4,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nnz)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(out) :: dweight(nnz), dbias(dim_out)

        integer :: k, col, start_, end_

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    dweight(start_:end_)= dweight(start_:end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                
                call mkl_scscmv('N', dim_in, dim_out, 1.0, matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), 0.0, dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backwards
    subroutine backward_convs(dy,x,dx,dweight, dbias, num_batch, csc_indptr,csc_indices, &
            csc_data, nnz,nd, dim_in,dim_out, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,nd, dim_out
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(out) :: dx(num_batch, dim_in)
        real*4,intent(in) :: x(num_batch, dim_in), dy(num_batch,dim_out), csc_data(nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(out) :: dweight(nd), dbias(dim_out)

        integer :: k, col, start_, end_,data_start_, data_end_, ifo

        dweight=0
        do k=1,num_batch
            if(do_wgrad) then
                !calculate dweight
                do col=1,dim_out
                    start_=csc_indptr(col)
                    end_=csc_indptr(col+1)-1
                    data_start_=modulo(start_-1, nd)+1
                    data_end_=modulo(end_-1, nd)+1
                    dweight(data_start_:data_end_)= dweight(data_start_:data_end_)+x(k,csc_indices(start_:end_))*dy(k,col)
                    enddo
            endif
            if(do_xgrad) then
                !calculate dx
                do ifo=1,nnz/nd
                    call mkl_scscmv('N', dim_in, dim_out, 1.0, matdescra, csc_data, csc_indices,&
                    csc_indptr(1:dim_in), csc_indptr(2:dim_in+1), dy(k,:), 0.0, dx(k,:))
                enddo
                
                call mkl_scscmv('N', dim_in, dim_out, 1.0, matdescra, csc_data, csc_indices,&
                csc_indptr(1:dim_out), csc_indptr(2:dim_out+1), dy(k,:), 0.0, dx(k,:))
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_convs
    end module lib