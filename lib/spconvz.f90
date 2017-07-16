!This is an automatically generated .f90 file.
!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    
    subroutine forward_general(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data, weight_indices,&
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
    end subroutine forward_general

    subroutine backward_general(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices,fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        complex*16,intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(inplace) dx, dweight, dbias

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
                    conjg(x_work), num_batch, zero, w_work, nfo)

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
                call zgemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    conjg(w_work), nfo, zero, x_work, num_batch)
                
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
    end subroutine backward_general

    subroutine forward1_general(x, y, bias, csc_indptr, csc_indices, fltr_data, weight_indices,&
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
    end subroutine forward1_general

    subroutine backward1_general(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices, fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz) 
        complex*16,intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, weight_indices, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(inplace) dx, dweight, dbias

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
                call zgemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    conjg(x_work), 1, zero, w_work, nfo)
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
                call zgemv('C', nfo, k, one,&
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
    end subroutine backward1_general
    
    subroutine forward_contiguous(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data,&
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
    end subroutine forward_contiguous

    subroutine backward_contiguous(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(num_batch, nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, num_batch, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(inplace) dx, dweight, dbias

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
                    conjg(x_work), num_batch, one, dweight, nfo)
                
            endif
            if(do_xgrad) then
                
                call zgemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    conjg(fltr_data), nfo, zero, x_work, num_batch)
                
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
    end subroutine backward_contiguous

    subroutine forward1_contiguous(x, y, bias, csc_indptr, csc_indices, fltr_data,&
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
    end subroutine forward1_contiguous

    subroutine backward1_contiguous(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        complex*16 :: x_work(nfi, max_nnz_row)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr, fltr_data
        !f2py intent(in) do_xgrad, do_wgrad, do_bgrad
        !f2py intent(in) nfi, nfo, nd, max_nnz_row, dim_in, dim_out, nnz
        !f2py intent(inplace) dx, dweight, dbias

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
                call zgemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    conjg(x_work), 1, one, dweight, nfo)
                
            endif
            if(do_xgrad) then
                
                !calculate dx
                call zgemv('C', nfo, k, one,&
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
    end subroutine backward1_contiguous
    
end module lib
