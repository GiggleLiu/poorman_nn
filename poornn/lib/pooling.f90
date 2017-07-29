!This is an f90 file automatically generated.
!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    subroutine forward_z(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        complex*16,intent(in) :: x(nfi, dim_in)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        complex*16,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            if(mode==0) then
                y(:,col)=maxval(abs(x(:,csc_indices(start_:end_-1))),2)
            else
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            endif
        enddo
    end subroutine forward_z

    subroutine backward_z(dy,x,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,mode)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,mode
        complex*16,intent(in) :: x(nfi, dim_in)
        complex*16,intent(in) :: dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*16,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, irow, ib
        integer,pointer :: rows(:)
        complex*16 :: y_work(nfi)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr
        !f2py intent(in) nfi, dim_in, dim_out, nnz
        !f2py intent(out) dx

        if(mode==0) dx=0

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            if(mode==0) then
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            else
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            endif
        enddo
    end subroutine backward_z
    subroutine forward_d(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        real*8,intent(in) :: x(nfi, dim_in)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*8,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            if(mode==0) then
                y(:,col)=maxval((x(:,csc_indices(start_:end_-1))),2)
            else
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            endif
        enddo
    end subroutine forward_d

    subroutine backward_d(dy,x,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,mode)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,mode
        real*8,intent(in) :: x(nfi, dim_in)
        real*8,intent(in) :: dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*8,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, irow, ib
        integer,pointer :: rows(:)
        real*8 :: y_work(nfi)
        real*8,parameter :: one=1D0
        real*8,parameter :: zero=0D0

        !f2py intent(in) x, dy, csc_indices, csc_indptr
        !f2py intent(in) nfi, dim_in, dim_out, nnz
        !f2py intent(out) dx

        if(mode==0) dx=0

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            if(mode==0) then
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc((x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            else
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            endif
        enddo
    end subroutine backward_d
    subroutine forward_s(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        real*4,intent(in) :: x(nfi, dim_in)
        integer,intent(in) :: csc_indices(nnz), csc_indptr(dim_out+1)
        real*4,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            if(mode==0) then
                y(:,col)=maxval((x(:,csc_indices(start_:end_-1))),2)
            else
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            endif
        enddo
    end subroutine forward_s

    subroutine backward_s(dy,x,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,mode)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,mode
        real*4,intent(in) :: x(nfi, dim_in)
        real*4,intent(in) :: dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*4,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, irow, ib
        integer,pointer :: rows(:)
        real*4 :: y_work(nfi)
        real*4,parameter :: one=1.0
        real*4,parameter :: zero=0.0

        !f2py intent(in) x, dy, csc_indices, csc_indptr
        !f2py intent(in) nfi, dim_in, dim_out, nnz
        !f2py intent(out) dx

        if(mode==0) dx=0

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            if(mode==0) then
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc((x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            else
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            endif
        enddo
    end subroutine backward_s
    end module lib