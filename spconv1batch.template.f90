!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    subroutine forward(x, y, bias, csc_indptr, csc_indices, fltr_data, weight_indices,&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        {{dtype}},intent(in) :: x(nfi, dim_in), bias(nfo)
        {{dtype}},intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        {{dtype}},intent(out) :: y(nfo, dim_out)

        {{dtype}} :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        integer :: start_, end_, col, ii, nnz_row, k
        {{dtype}},parameter :: one={{dtype_one}}
        real*8 :: t0,t1,t2,dt1=0,dt2=0
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, weight_indices, bias
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

            call cpu_time(t0)
            !prepair work space by taking rows in x.
            do ii=1,nnz_row
                x_work(:,ii)=x(:,csc_indices(start_+ii-1))
                w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
            enddo
            call cpu_time(t1)
            call {{dtype_token}}gemv('N', nfo, k, one, w_work, nfo,&
            x_work, 1, one, y(:,col), 1)
            call cpu_time(t2)
            dt1=dt1+t1-t0
            dt2=dt2+t2-t1
        enddo
        print*,dt1,dt2
    end subroutine forward

    subroutine backward(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,weight_indices,fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        {{dtype}},intent(in) :: x(nfi, dim_in), dy(nfo, dim_out),&
            fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1), weight_indices(nnz)
        {{dtype}},intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        {{dtype}} :: x_work(nfi, max_nnz_row), w_work(nfo, nfi, max_nnz_row)
        {{dtype}},parameter :: one={{dtype_one}}
        {{dtype}},parameter :: zero={{dtype_zero}}

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
                    x_work(:,:,ii)=x(:,:,csc_indices(start_+ii-1))
                enddo

                !calculate dweight
                call {{dtype_token}}gemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
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
                call {{dtype_token}}gemm('N', 'N', num_batch, k, nfo, one,&
                    {%ifequal dtype "complex*16"%}conjg(dy(:,:,col)){%else%}dy(:,:,col){%endifequal%}, num_batch,&
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
