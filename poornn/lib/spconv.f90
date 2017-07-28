!This is an f90 file automatically generated.
!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    subroutine forward_generalz(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data, weight_indices,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        complex*16,intent(in) :: x(num_batch, nfi, dim_in), bias(nfo)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        complex*16,intent(out) :: y(num_batch, nfo, dim_out)

        complex*16 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, bias, weight_indices
        !f2py intent(in) nfi, nfo, num_batch, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(:,ii,:)=bias(ii)
        enddo

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
            enddo
            call zgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                w_work, nfo, one, y(:,:,col), num_batch)
        enddo
    end subroutine forward_generalz

    subroutine backward_generalz(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        complex*16,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                call zgemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    x_work, num_batch, zero, w_work, nfo)

                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                endif
            if(do_xgrad) then
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                !calculate dx
                call zgemm('N', 'N', num_batch, k, nfo, one, (dy(:,:,col)), num_batch,&
                    w_work, nfo, zero, x_work, num_batch)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,:,row)=dx(:,:,row)+x_work(:,:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(sum(dy,1),2)
        endif
    end subroutine backward_generalz

    subroutine forward1_generalz(x, y, bias, csc_indptr, csc_indices, fltr_data, weight_indices,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        complex*16,intent(in) :: x(nfi, dim_in), bias(nfo)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz) 
        complex*16,intent(out) :: y(nfo, dim_out)

        complex*16 :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, weight_indices,  bias
        !f2py intent(in) nfi, nfo, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(ii,:)=bias(ii)
        enddo
        
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
            enddo
            call zgemv('N', nfo, k, one, fltr_data, nfo,&
            x_work, 1, one, y(:,col), 1)
        enddo
    end subroutine forward1_generalz

    subroutine backward1_generalz(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices, fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz) 
        complex*16,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                w_work=0
                !call zgerc(nfo, k, one, dy(:,col), 1,&
                !    x_work, 1, w_work, nfo)
                call zgemm('N', 'N', nfo, k, 1, one, (dy(:,col)), nfo,&
                    x_work, 1, zero, w_work, nfo)
                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                endif
            if(do_xgrad) then
                
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                
                !calculate dx
                call zgemv('T', nfo, k, one,&
                    w_work, nfo, (dy(:,col)), 1, zero, x_work, 1)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,row)=dx(:,row)+x_work(:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(dy,2)
        endif
    end subroutine backward1_generalz
    subroutine forward_contiguousz(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        complex*16,intent(in) :: x(num_batch, nfi, dim_in), bias(nfo)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(out) :: y(num_batch, nfo, dim_out)

        complex*16 :: x_work(num_batch, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, bias
        !f2py intent(in) nfi, nfo, num_batch, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(:,ii,:)=bias(ii)
        enddo

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                
            enddo
            call zgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                fltr_data, nfo, one, y(:,:,col), num_batch)
        enddo
    end subroutine forward_contiguousz

    subroutine backward_contiguousz(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(num_batch, nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                call zgemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    x_work, num_batch, one, dweight, nfo)
                endif
            if(do_xgrad) then
                call zgemm('N', 'N', num_batch, k, nfo, one, (dy(:,:,col)), num_batch,&
                    fltr_data, nfo, zero, x_work, num_batch)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,:,row)=dx(:,:,row)+x_work(:,:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(sum(dy,1),2)
        endif
    end subroutine backward_contiguousz

    subroutine forward1_contiguousz(x, y, bias, csc_indptr, csc_indices, fltr_data,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        complex*16,intent(in) :: x(nfi, dim_in), bias(nfo)
        complex*16,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(out) :: y(nfo, dim_out)

        complex*16 :: x_work(nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data,bias
        !f2py intent(in) nfi, nfo, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(ii,:)=bias(ii)
        enddo
        
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                
            enddo
            call zgemv('N', nfo, k, one, fltr_data, nfo,&
            x_work, 1, one, y(:,col), 1)
        enddo
    end subroutine forward1_contiguousz

    subroutine backward1_contiguousz(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                !Q: slow!!!!
                !call zgerc(nfo, k, one, dy(:,col), 1,& 
                !    conjg(x_work), 1, dweight, nfo)
                call zgemm('N', 'N', nfo, k, 1, one, (dy(:,col)), nfo,&
                    x_work, 1, one, dweight, nfo)
                endif
            if(do_xgrad) then
                
                !calculate dx
                call zgemv('T', nfo, k, one,&
                    fltr_data, nfo, (dy(:,col)), 1, zero, x_work, 1)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,row)=dx(:,row)+x_work(:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(dy,2)
        endif
    end subroutine backward1_contiguousz
    subroutine forward_generald(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data, weight_indices,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*8,intent(in) :: x(num_batch, nfi, dim_in), bias(nfo)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        real*8,intent(out) :: y(num_batch, nfo, dim_out)

        real*8 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*8,parameter :: one=1D0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, bias, weight_indices
        !f2py intent(in) nfi, nfo, num_batch, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(:,ii,:)=bias(ii)
        enddo

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
            enddo
            call dgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                w_work, nfo, one, y(:,:,col), num_batch)
        enddo
    end subroutine forward_generald

    subroutine backward_generald(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        real*8,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*8 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        real*8,parameter :: one=1D0
        real*8,parameter :: zero=0D0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                call dgemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    x_work, num_batch, zero, w_work, nfo)

                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                endif
            if(do_xgrad) then
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                !calculate dx
                call dgemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    w_work, nfo, zero, x_work, num_batch)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,:,row)=dx(:,:,row)+x_work(:,:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(sum(dy,1),2)
        endif
    end subroutine backward_generald

    subroutine forward1_generald(x, y, bias, csc_indptr, csc_indices, fltr_data, weight_indices,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*8,intent(in) :: x(nfi, dim_in), bias(nfo)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz) 
        real*8,intent(out) :: y(nfo, dim_out)

        real*8 :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*8,parameter :: one=1D0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, weight_indices,  bias
        !f2py intent(in) nfi, nfo, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(ii,:)=bias(ii)
        enddo
        
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
            enddo
            call dgemv('N', nfo, k, one, fltr_data, nfo,&
            x_work, 1, one, y(:,col), 1)
        enddo
    end subroutine forward1_generald

    subroutine backward1_generald(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices, fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz) 
        real*8,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*8 :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        real*8,parameter :: one=1D0
        real*8,parameter :: zero=0D0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                w_work=0
                !call dger(nfo, k, one, dy(:,col), 1,&
                !    x_work, 1, w_work, nfo)
                call dgemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    x_work, 1, zero, w_work, nfo)
                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                endif
            if(do_xgrad) then
                
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                
                !calculate dx
                call dgemv('T', nfo, k, one,&
                    w_work, nfo, dy(:,col), 1, zero, x_work, 1)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,row)=dx(:,row)+x_work(:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(dy,2)
        endif
    end subroutine backward1_generald
    subroutine forward_contiguousd(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*8,intent(in) :: x(num_batch, nfi, dim_in), bias(nfo)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(out) :: y(num_batch, nfo, dim_out)

        real*8 :: x_work(num_batch, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*8,parameter :: one=1D0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, bias
        !f2py intent(in) nfi, nfo, num_batch, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(:,ii,:)=bias(ii)
        enddo

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                
            enddo
            call dgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                fltr_data, nfo, one, y(:,:,col), num_batch)
        enddo
    end subroutine forward_contiguousd

    subroutine backward_contiguousd(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*8 :: x_work(num_batch, nfi, max_nnz_row)
        real*8,parameter :: one=1D0
        real*8,parameter :: zero=0D0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                call dgemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    x_work, num_batch, one, dweight, nfo)
                endif
            if(do_xgrad) then
                call dgemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    fltr_data, nfo, zero, x_work, num_batch)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,:,row)=dx(:,:,row)+x_work(:,:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(sum(dy,1),2)
        endif
    end subroutine backward_contiguousd

    subroutine forward1_contiguousd(x, y, bias, csc_indptr, csc_indices, fltr_data,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*8,intent(in) :: x(nfi, dim_in), bias(nfo)
        real*8,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(out) :: y(nfo, dim_out)

        real*8 :: x_work(nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*8,parameter :: one=1D0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data,bias
        !f2py intent(in) nfi, nfo, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(ii,:)=bias(ii)
        enddo
        
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                
            enddo
            call dgemv('N', nfo, k, one, fltr_data, nfo,&
            x_work, 1, one, y(:,col), 1)
        enddo
    end subroutine forward1_contiguousd

    subroutine backward1_contiguousd(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*8 :: x_work(nfi, max_nnz_row)
        real*8,parameter :: one=1D0
        real*8,parameter :: zero=0D0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                !Q: slow!!!!
                !call dger(nfo, k, one, dy(:,col), 1,& 
                !    x_work, 1, dweight, nfo)
                call dgemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    x_work, 1, one, dweight, nfo)
                endif
            if(do_xgrad) then
                
                !calculate dx
                call dgemv('T', nfo, k, one,&
                    fltr_data, nfo, dy(:,col), 1, zero, x_work, 1)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,row)=dx(:,row)+x_work(:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(dy,2)
        endif
    end subroutine backward1_contiguousd
    subroutine forward_generals(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data, weight_indices,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*4,intent(in) :: x(num_batch, nfi, dim_in), bias(nfo)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        real*4,intent(out) :: y(num_batch, nfo, dim_out)

        real*4 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*4,parameter :: one=1.0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, bias, weight_indices
        !f2py intent(in) nfi, nfo, num_batch, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(:,ii,:)=bias(ii)
        enddo

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
            enddo
            call sgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                w_work, nfo, one, y(:,:,col), num_batch)
        enddo
    end subroutine forward_generals

    subroutine backward_generals(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        real*4,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*4 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        real*4,parameter :: one=1.0
        real*4,parameter :: zero=0.0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                call sgemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    x_work, num_batch, zero, w_work, nfo)

                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                endif
            if(do_xgrad) then
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                !calculate dx
                call sgemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    w_work, nfo, zero, x_work, num_batch)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,:,row)=dx(:,:,row)+x_work(:,:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(sum(dy,1),2)
        endif
    end subroutine backward_generals

    subroutine forward1_generals(x, y, bias, csc_indptr, csc_indices, fltr_data, weight_indices,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*4,intent(in) :: x(nfi, dim_in), bias(nfo)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz) 
        real*4,intent(out) :: y(nfo, dim_out)

        real*4 :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*4,parameter :: one=1.0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, weight_indices,  bias
        !f2py intent(in) nfi, nfo, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(ii,:)=bias(ii)
        enddo
        
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
            enddo
            call sgemv('N', nfo, k, one, fltr_data, nfo,&
            x_work, 1, one, y(:,col), 1)
        enddo
    end subroutine forward1_generals

    subroutine backward1_generals(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices, fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz) 
        real*4,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*4 :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        real*4,parameter :: one=1.0
        real*4,parameter :: zero=0.0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                w_work=0
                !call sger(nfo, k, one, dy(:,col), 1,&
                !    x_work, 1, w_work, nfo)
                call sgemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    x_work, 1, zero, w_work, nfo)
                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                endif
            if(do_xgrad) then
                
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                
                !calculate dx
                call sgemv('T', nfo, k, one,&
                    w_work, nfo, dy(:,col), 1, zero, x_work, 1)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,row)=dx(:,row)+x_work(:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(dy,2)
        endif
    end subroutine backward1_generals
    subroutine forward_contiguouss(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*4,intent(in) :: x(num_batch, nfi, dim_in), bias(nfo)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(out) :: y(num_batch, nfo, dim_out)

        real*4 :: x_work(num_batch, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*4,parameter :: one=1.0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, bias
        !f2py intent(in) nfi, nfo, num_batch, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(:,ii,:)=bias(ii)
        enddo

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                
            enddo
            call sgemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                fltr_data, nfo, one, y(:,:,col), num_batch)
        enddo
    end subroutine forward_contiguouss

    subroutine backward_contiguouss(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*4 :: x_work(num_batch, nfi, max_nnz_row)
        real*4,parameter :: one=1.0
        real*4,parameter :: zero=0.0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                call sgemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    x_work, num_batch, one, dweight, nfo)
                endif
            if(do_xgrad) then
                call sgemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    fltr_data, nfo, zero, x_work, num_batch)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,:,row)=dx(:,:,row)+x_work(:,:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(sum(dy,1),2)
        endif
    end subroutine backward_contiguouss

    subroutine forward1_contiguouss(x, y, bias, csc_indptr, csc_indices, fltr_data,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        real*4,intent(in) :: x(nfi, dim_in), bias(nfo)
        real*4,intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(out) :: y(nfo, dim_out)

        real*4 :: x_work(nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        real*4,parameter :: one=1.0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data,bias
        !f2py intent(in) nfi, nfo, max_nnz_row, nnz, dim_out, nd, dim_in
        !f2py intent(out) y

        do ii=1,nfo
            y(ii,:)=bias(ii)
        enddo
        
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                
            enddo
            call sgemv('N', nfo, k, one, fltr_data, nfo,&
            x_work, 1, one, y(:,col), 1)
        enddo
    end subroutine forward1_contiguouss

    subroutine backward1_contiguouss(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(out) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*4 :: x_work(nfi, max_nnz_row)
        real*4,parameter :: one=1.0
        real*4,parameter :: zero=0.0

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(out) dx, dweight, dbias

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            nnz_row=end_-start_
            k=nfi*nnz_row

            if(do_wgrad) then
                !prepair work space by taking rows in x
                do ii=1,nnz_row
                    x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                !Q: slow!!!!
                !call sger(nfo, k, one, dy(:,col), 1,& 
                !    x_work, 1, dweight, nfo)
                call sgemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    x_work, 1, one, dweight, nfo)
                endif
            if(do_xgrad) then
                
                !calculate dx
                call sgemv('T', nfo, k, one,&
                    fltr_data, nfo, dy(:,col), 1, zero, x_work, 1)
                !extract rows
                do ii=1,nnz_row
                    row=csc_indices(start_+ii-1)
                    dx(:,row)=dx(:,row)+x_work(:,ii)
                enddo
            endif
        enddo
        if(do_bgrad) then
            !calculate dbias
            dbias=dbias+sum(dy,2)
        endif
    end subroutine backward1_contiguouss
    end module lib