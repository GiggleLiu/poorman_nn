!This is an f90 file automatically generated.
!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    !mode = 0: max real.
    !mode = 1: max abs.
    !mode = 2: min real.
    !mode = 3: min abs.
    !mode = 4: mean pooling.
    subroutine forward_z(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        complex*16,intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*16,intent(out) :: y(nfi, dim_out)

        integer,pointer :: rows(:)
        integer :: start_, end_, col, ib, irow
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            select case (mode)
            case (0)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(real(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
                case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (2)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(real(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
                case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (4)
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
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

            select case (mode)
            case (0)
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(real(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (2)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(real(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (4)
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
        enddo
    end subroutine backward_z
    !mode = 0: max real.
    !mode = 1: max abs.
    !mode = 2: min real.
    !mode = 3: min abs.
    !mode = 4: mean pooling.
    subroutine forward_c(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        complex*8,intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*8,intent(out) :: y(nfi, dim_out)

        integer,pointer :: rows(:)
        integer :: start_, end_, col, ib, irow
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            select case (mode)
            case (0)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(real(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
                case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (2)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(real(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
                case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (4)
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
        enddo
    end subroutine forward_c

    subroutine backward_c(dy,x,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,mode)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi,mode
        complex*8,intent(in) :: x(nfi, dim_in)
        complex*8,intent(in) :: dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*8,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, irow, ib
        integer,pointer :: rows(:)
        complex*8 :: y_work(nfi)
        complex*8,parameter :: one=cmplx(1.0,0.0)
        complex*8,parameter :: zero=cmplx(0.0,0.0)

        !f2py intent(in) x, dy, csc_indices, csc_indptr
        !f2py intent(in) nfi, dim_in, dim_out, nnz
        !f2py intent(out) dx

        if(mode==0) dx=0

        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            select case (mode)
            case (0)
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(real(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (2)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(real(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (4)
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
        enddo
    end subroutine backward_c
    !mode = 0: max real.
    !mode = 1: max abs.
    !mode = 2: min real.
    !mode = 3: min abs.
    !mode = 4: mean pooling.
    subroutine forward_d(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        real*8,intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*8,intent(out) :: y(nfi, dim_out)

        integer,pointer :: rows(:)
        integer :: start_, end_, col, ib, irow
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            select case (mode)
            case (0)
                y(:,col)=maxval(x(:,csc_indices(start_:end_-1)),2)
                case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (2)
                y(:,col)=minval(x(:,csc_indices(start_:end_-1)),2)
                case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (4)
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
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

            select case (mode)
            case (0)
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc((x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (2)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc((x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (4)
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
        enddo
    end subroutine backward_d
    !mode = 0: max real.
    !mode = 1: max abs.
    !mode = 2: min real.
    !mode = 3: min abs.
    !mode = 4: mean pooling.
    subroutine forward_s(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, mode)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi,mode
        real*4,intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*4,intent(out) :: y(nfi, dim_out)

        integer,pointer :: rows(:)
        integer :: start_, end_, col, ib, irow
        !f2py intent(in) x, csc_indices, csc_indptr
        !f2py intent(in) nfi, nnz, dim_out, dim_in
        !f2py intent(out) y

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            !prepair work space by taking rows in x.
            select case (mode)
            case (0)
                y(:,col)=maxval(x(:,csc_indices(start_:end_-1)),2)
                case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (2)
                y(:,col)=minval(x(:,csc_indices(start_:end_-1)),2)
                case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    y(ib,col)=x(ib,rows(irow))
                enddo
            case (4)
                y(:,col)=sum(x(:,csc_indices(start_:end_-1)),2)/(end_-start_)
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
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

            select case (mode)
            case (0)
                !prepair work space by taking rows in x.
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc((x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (1)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=maxloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (2)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc((x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (3)
                rows=>csc_indices(start_:end_-1)
                do ib=1,nfi
                    irow=minloc(abs(x(ib,rows)),1)
                    dx(ib,rows(irow))=dy(ib,col)
                enddo
            case (4)
                y_work=dy(:,col)/(end_-start_)
                do irow=start_,end_-1
                    dx(:,csc_indices(irow))=y_work
                enddo
            case default
                print*,'Error: Pooling mode not exist!'
                stop 1
            endselect
        enddo
    end subroutine backward_s
    end module lib