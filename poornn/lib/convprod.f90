!This is an f90 file automatically generated.
module lib
    contains
    subroutine forward_z(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, powers, nd)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nd
        complex*16,intent(in) :: x(nfi, dim_in)
        complex*16,intent(in):: powers(nd)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*16,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col, irow

        y=1
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            do irow=1,end_-start_
                y(:,col)=y(:,col)*x(:,csc_indices(start_+irow-1))**powers(irow)
            enddo
        enddo
    end subroutine forward_z

    subroutine backward_z(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,powers,nd)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi, nd
        complex*16,intent(in) :: x(nfi, dim_in)
        complex*16,intent(in):: powers(nd)
        complex*16,intent(in) :: y(nfi, dim_out), dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*16,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, ib
        integer,pointer :: rows(:)

        dx=0
        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            rows=>csc_indices(start_:end_-1)
            do ib=1,nfi
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)*powers/x(ib,rows)
            enddo
        enddo
    end subroutine backward_z
    subroutine forward_c(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, powers, nd)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nd
        complex*8,intent(in) :: x(nfi, dim_in)
        complex*8,intent(in):: powers(nd)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*8,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col, irow

        y=1
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            do irow=1,end_-start_
                y(:,col)=y(:,col)*x(:,csc_indices(start_+irow-1))**powers(irow)
            enddo
        enddo
    end subroutine forward_c

    subroutine backward_c(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,powers,nd)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi, nd
        complex*8,intent(in) :: x(nfi, dim_in)
        complex*8,intent(in):: powers(nd)
        complex*8,intent(in) :: y(nfi, dim_out), dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*8,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, ib
        integer,pointer :: rows(:)

        dx=0
        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            rows=>csc_indices(start_:end_-1)
            do ib=1,nfi
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)*powers/x(ib,rows)
            enddo
        enddo
    end subroutine backward_c
    subroutine forward_d(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, powers, nd)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nd
        real*8,intent(in) :: x(nfi, dim_in)
        real*8,intent(in):: powers(nd)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*8,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col, irow

        y=1
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            do irow=1,end_-start_
                y(:,col)=y(:,col)*x(:,csc_indices(start_+irow-1))**powers(irow)
            enddo
        enddo
    end subroutine forward_d

    subroutine backward_d(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,powers,nd)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi, nd
        real*8,intent(in) :: x(nfi, dim_in)
        real*8,intent(in):: powers(nd)
        real*8,intent(in) :: y(nfi, dim_out), dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*8,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, ib
        integer,pointer :: rows(:)

        dx=0
        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            rows=>csc_indices(start_:end_-1)
            do ib=1,nfi
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)*powers/x(ib,rows)
            enddo
        enddo
    end subroutine backward_d
    subroutine forward_s(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi, powers, nd)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi, nd
        real*4,intent(in) :: x(nfi, dim_in)
        real*4,intent(in):: powers(nd)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*4,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col, irow

        y=1
        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            do irow=1,end_-start_
                y(:,col)=y(:,col)*x(:,csc_indices(start_+irow-1))**powers(irow)
            enddo
        enddo
    end subroutine forward_s

    subroutine backward_s(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi,powers,nd)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi, nd
        real*4,intent(in) :: x(nfi, dim_in)
        real*4,intent(in):: powers(nd)
        real*4,intent(in) :: y(nfi, dim_out), dy(nfi, dim_out)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*4,intent(out) :: dx(nfi, dim_in)

        integer :: start_, end_, col, ib
        integer,pointer :: rows(:)

        dx=0
        do col=1,dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)

            rows=>csc_indices(start_:end_-1)
            do ib=1,nfi
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)*powers/x(ib,rows)
            enddo
        enddo
    end subroutine backward_s
    end module lib