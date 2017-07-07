!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    subroutine forward(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data, weight_indices,&
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
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, weight_indices, bias
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
    end subroutine forward

    subroutine backward(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices,fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        real*4,intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        real*4 :: x_work(num_batch, nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        real*4,parameter :: one=1.0
        real*4,parameter :: zero=0.0

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
                call sgemm('N', 'N', num_batch, k, nfo, one,&
                    dy(:,:,col), num_batch,&
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
    end subroutine backward
end module lib
