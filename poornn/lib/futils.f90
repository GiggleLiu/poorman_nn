!This is an f90 file automatically generated.
subroutine fsign_z(x, nx, s)
    implicit none
    integer,intent(in) :: nx
    complex*16,intent(in) :: x(nx)
    complex*16,intent(out) :: s(nx)

    complex*16 :: xi
    integer :: i

    do i=1,nx
        xi = x(i)
        if(xi==0) then
            s(i) = 0
        else
            s(i) = xi/abs(xi)
        endif
    enddo
end subroutine fsign_z
subroutine fsign_c(x, nx, s)
    implicit none
    integer,intent(in) :: nx
    complex*8,intent(in) :: x(nx)
    complex*8,intent(out) :: s(nx)

    complex*8 :: xi
    integer :: i

    do i=1,nx
        xi = x(i)
        if(xi==0) then
            s(i) = 0
        else
            s(i) = xi/abs(xi)
        endif
    enddo
end subroutine fsign_c
subroutine fsign_d(x, nx, s)
    implicit none
    integer,intent(in) :: nx
    real*8,intent(in) :: x(nx)
    real*8,intent(out) :: s(nx)

    real*8 :: xi
    integer :: i

    do i=1,nx
        xi = x(i)
        if(xi==0) then
            s(i) = 0
        else
            s(i) = xi/abs(xi)
        endif
    enddo
end subroutine fsign_d
subroutine fsign_s(x, nx, s)
    implicit none
    integer,intent(in) :: nx
    real*4,intent(in) :: x(nx)
    real*4,intent(out) :: s(nx)

    real*4 :: xi
    integer :: i

    do i=1,nx
        xi = x(i)
        if(xi==0) then
            s(i) = 0
        else
            s(i) = xi/abs(xi)
        endif
    enddo
end subroutine fsign_s
