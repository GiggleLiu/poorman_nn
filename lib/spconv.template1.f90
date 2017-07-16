!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    {%for version in version_list%}
    subroutine forward_{{version}}(x, y, bias, num_batch, csc_indptr, csc_indices, fltr_data,{%ifequal version "general"%} weight_indices,{%endifequal%}&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        {{dtype}},intent(in) :: x(num_batch, nfi, dim_in), bias(nfo)
        {{dtype}},intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1){%ifequal version "general"%}, weight_indices(nnz){%endifequal%}
        {{dtype}},intent(out) :: y(num_batch, nfo, dim_out)

        {{dtype}} :: x_work(num_batch, nfi, max_nnz_row){%ifequal version "general"%}, w_work(nfo, nfi, max_nnz_row){%endifequal%}
        integer :: start_, end_, col, ii, nnz_row, k
        {{dtype}},parameter :: one={{dtype_one}}
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data, bias{%ifequal version "general"%}, weight_indices{%endifequal%}
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
                {%ifequal version "general"%}w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1)){%endifequal%}
            enddo
            call {{dtype_token}}gemm('N', 'T', num_batch, nfo, k, one, x_work, num_batch,&
                {%ifequal version "contiguous"%}fltr_data{%else%}w_work{%endifequal%}, nfo, one, y(:,:,col), num_batch)
        enddo
    end subroutine forward_{{version}}

    subroutine backward_{{version}}(dy,x,dx,dweight,dbias,csc_indptr,csc_indices{%ifequal version "general"%},weight_indices{%endifequal%},fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, num_batch, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: num_batch, nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        {{dtype}},intent(in) :: x(num_batch, nfi, dim_in), dy(num_batch, nfo, dim_out),&
            fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1){%ifequal version "general"%}, weight_indices(nnz){%endifequal%}
        {{dtype}},intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(num_batch, nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        {{dtype}} :: x_work(num_batch, nfi, max_nnz_row){%ifequal version "general"%}, w_work(nfo, nfi, max_nnz_row){%endifequal%}
        {{dtype}},parameter :: one={{dtype_one}}
        {{dtype}},parameter :: zero={{dtype_zero}}

        !f2py intent(in) x, dy, csc_indices, csc_indptr{%ifequal version "general"%}, weight_indices{%endifequal%}, fltr_data
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
                {%ifequal version "general"%}
                call {{dtype_token}}gemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    {%if is_complex%}conjg(x_work){%else%}x_work{%endif%}, num_batch, zero, w_work, nfo)

                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                {%else%}
                call {{dtype_token}}gemm('T', 'N', nfo, k, num_batch, one, dy(:,:,col), num_batch,&
                    {%if is_complex%}conjg(x_work){%else%}x_work{%endif%}, num_batch, one, dweight, nfo)
                {%endifequal%}
            endif
            if(do_xgrad) then
                {%ifequal version "general"%}
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                !calculate dx
                call {{dtype_token}}gemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    {%if is_complex%}conjg(w_work){%else%}w_work{%endif%}, nfo, zero, x_work, num_batch)
                {%else%}
                call {{dtype_token}}gemm('N', 'N', num_batch, k, nfo, one, dy(:,:,col), num_batch,&
                    {%if is_complex%}conjg(fltr_data){%else%}fltr_data{%endif%}, nfo, zero, x_work, num_batch)
                {%endifequal%}
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
    end subroutine backward_{{version}}

    subroutine forward1_{{version}}(x, y, bias, csc_indptr, csc_indices, fltr_data{%ifequal version "general"%}, weight_indices{%endifequal%},&
            nnz, dim_in, dim_out, nfi, nfo, nd, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nfo, max_nnz_row, nd
        {{dtype}},intent(in) :: x(nfi, dim_in), bias(nfo)
        {{dtype}},intent(in) :: fltr_data(nfo, nfi, nd)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1){%ifequal version "general"%}, weight_indices(nnz) {%endifequal%}
        {{dtype}},intent(out) :: y(nfo, dim_out)

        {{dtype}} :: x_work(nfi, max_nnz_row){%ifequal version "general"%}, w_work(nfo, nfi, max_nnz_row){%endifequal%}
        integer :: start_, end_, col, ii, nnz_row, k
        {{dtype}},parameter :: one={{dtype_one}}
        !f2py intent(in) x, csc_indices, csc_indptr, fltr_data,{%ifequal version "general"%} weight_indices,  {%endifequal%}bias
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
                {%ifequal version "general"%}w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1)){%endifequal%}
            enddo
            call {{dtype_token}}gemv('N', nfo, k, one, {%ifequal version general%}w_work{%else%}fltr_data{%endifequal%}, nfo,&
            x_work, 1, one, y(:,col), 1)
        enddo
    end subroutine forward1_{{version}}

    subroutine backward1_{{version}}(dy,x,dx,dweight,dbias,csc_indptr,csc_indices,{%ifequal version "general"%}weight_indices, {%endifequal%}fltr_data,bias,&
            nnz,dim_in,dim_out,nfi,nfo, nd, do_xgrad, do_wgrad, do_bgrad, max_nnz_row)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,nfo,nd,max_nnz_row
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        {{dtype}},intent(in) :: x(nfi, dim_in), dy(nfo, dim_out), fltr_data(nfo,nfi,nd), bias(nfo)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1){%ifequal version "general"%}, weight_indices(nnz) {%endifequal%}
        {{dtype}},intent(inout) :: dweight(nfo, nfi, nd), dbias(nfo), dx(nfi, dim_in)

        integer :: start_, end_, k, col, ii, row, nnz_row
        {{dtype}} :: x_work(nfi, max_nnz_row){%ifequal version "general"%}, w_work(nfo, nfi, max_nnz_row){%endifequal%}
        {{dtype}},parameter :: one={{dtype_one}}
        {{dtype}},parameter :: zero={{dtype_zero}}

        !f2py intent(in) x, dy, csc_indices, csc_indptr, {%ifequal version "general"%}weight_indices, {%endifequal%}fltr_data
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
                {%ifequal version "general"%}
                w_work=0
                !call {{dtype_token}}ger{%if is_complex%}c{%endif%}(nfo, k, one, dy(:,col), 1,&
                !    x_work, 1, w_work, nfo)
                call {{dtype_token}}gemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    {%if is_complex%}conjg(x_work){%else%}x_work{%endif%}, 1, zero, w_work, nfo)
                !extract rows
                do ii=1,nnz_row
                    row=weight_indices(start_+ii-1)
                    dweight(:,:,row)=dweight(:,:,row)+w_work(:,:,ii)
                enddo
                {%else%}
                !Q: slow!!!!
                !call {{dtype_token}}ger{%if is_complex%}c{%endif%}(nfo, k, one, dy(:,col), 1,& 
                !    {%if is_complex%}conjg(x_work){%else%}x_work{%endif%}, 1, dweight, nfo)
                call {{dtype_token}}gemm('N', 'N', nfo, k, 1, one, dy(:,col), nfo,&
                    {%if is_complex%}conjg(x_work){%else%}x_work{%endif%}, 1, one, dweight, nfo)
                {%endifequal%}
            endif
            if(do_xgrad) then
                {%ifequal version "general"%}
                !prepair work space by taking rows in weight
                do ii=1,nnz_row
                    w_work(:,:,ii)=fltr_data(:,:,weight_indices(start_+ii-1))
                enddo
                {%endifequal%}
                !calculate dx
                call {{dtype_token}}gemv({%if is_complex%}'C'{%else%}'T'{%endif%}, nfo, k, one,&
                    {%ifequal version "general"%}w_work{%else%}fltr_data{%endifequal%}, nfo, dy(:,col), 1, zero, x_work, 1)
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
    end subroutine backward1_{{version}}
    {%endfor%}
end module lib
