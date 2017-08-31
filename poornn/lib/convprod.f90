!This is an f90 file automatically generated.
module lib
    contains
    subroutine forward_z(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi
        complex*16,intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        complex*16,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            y(:,col)=product(x(:,csc_indices(start_:end_-1)),2)
        enddo
    end subroutine forward_z

    subroutine backward_z(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi
        complex*16,intent(in) :: x(nfi, dim_in)
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
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)/x(ib,rows)
            enddo
        enddo
    end subroutine backward_z
    subroutine forward_d(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi
        real*8,intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*8,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            y(:,col)=product(x(:,csc_indices(start_:end_-1)),2)
        enddo
    end subroutine forward_d

    subroutine backward_d(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi
        real*8,intent(in) :: x(nfi, dim_in)
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
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)/x(ib,rows)
            enddo
        enddo
    end subroutine backward_d
    subroutine forward_s(x, y, csc_indptr, csc_indices, nnz, dim_in, dim_out, nfi)
        implicit none
        integer,intent(in) :: nnz, dim_in, dim_out, nfi
        real*4,intent(in) :: x(nfi, dim_in)
        integer,intent(in),target :: csc_indices(nnz)
        integer,intent(in) :: csc_indptr(dim_out+1)
        real*4,intent(out) :: y(nfi, dim_out)

        integer :: start_, end_, col

        do col=1, dim_out
            start_=csc_indptr(col)
            end_=csc_indptr(col+1)
            y(:,col)=product(x(:,csc_indices(start_:end_-1)),2)
        enddo
    end subroutine forward_s

    subroutine backward_s(dy,x,y,dx,csc_indptr,csc_indices, nnz,dim_in,dim_out,nfi)
        implicit none
        integer,intent(in) :: nnz,dim_in,dim_out,nfi
        real*4,intent(in) :: x(nfi, dim_in)
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
                dx(ib,rows) = dx(ib,rows)+dy(ib,col)*y(ib,col)/x(ib,rows)
            enddo
        enddo
    end subroutine backward_s
    end module lib