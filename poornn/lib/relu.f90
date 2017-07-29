!This is an f90 file automatically generated.
!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
module lib
    contains
    subroutine forward_z(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        complex*16,intent(in) :: leak
        complex*16,intent(in) :: x(dim_in)
        complex*16,intent(out) :: y(dim_in)
        integer :: i
        complex*16 :: xi
        do i=1,dim_in
            xi=x(i)
            if(aimag(xi)<0 .or. real(xi)<0) then
                y(i)=leak*xi
            else
                y(i)=xi
            endif
        enddo
    end subroutine forward_z

    subroutine backward_z(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        complex*16,intent(in) :: leak
        complex*16,intent(in) :: x(dim_in)
        complex*16,intent(in) :: dy(dim_in)
        complex*16,intent(out) :: dx(dim_in)
        complex*16 :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if(aimag(xi)<0 .or. real(xi)<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dy(i)
            endif
        enddo
    end subroutine backward_z
    subroutine forward_d(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        real*8,intent(in) :: leak
        real*8,intent(in) :: x(dim_in)
        real*8,intent(out) :: y(dim_in)
        integer :: i
        real*8 :: xi
        do i=1,dim_in
            xi=x(i)
            if(xi<0) then
                y(i)=leak*xi
            else
                y(i)=xi
            endif
        enddo
    end subroutine forward_d

    subroutine backward_d(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        real*8,intent(in) :: leak
        real*8,intent(in) :: x(dim_in)
        real*8,intent(in) :: dy(dim_in)
        real*8,intent(out) :: dx(dim_in)
        real*8 :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if(xi<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dy(i)
            endif
        enddo
    end subroutine backward_d
    subroutine forward_s(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        real*4,intent(in) :: leak
        real*4,intent(in) :: x(dim_in)
        real*4,intent(out) :: y(dim_in)
        integer :: i
        real*4 :: xi
        do i=1,dim_in
            xi=x(i)
            if(xi<0) then
                y(i)=leak*xi
            else
                y(i)=xi
            endif
        enddo
    end subroutine forward_s

    subroutine backward_s(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        real*4,intent(in) :: leak
        real*4,intent(in) :: x(dim_in)
        real*4,intent(in) :: dy(dim_in)
        real*4,intent(out) :: dx(dim_in)
        real*4 :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if(xi<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dy(i)
            endif
        enddo
    end subroutine backward_s
    
end module lib