!This is an f90 file automatically generated.
!orders: conv_dim_out/in, feature_dim_out/in, batch_dim
!version: 1 -> real-imagine seperate, otherwise real only
module lib
    contains
    subroutine forward_rz(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        real*8,intent(in) :: leak
        complex*16,intent(in) :: x(dim_in)
        complex*16,intent(out) :: y(dim_in)
        integer :: i
        complex*16 :: xi
        do i=1,dim_in
            xi=x(i)
            if(real(xi)<0) then
                y(i)=leak*xi
            else
                y(i)=xi
            endif
            
        enddo
    end subroutine forward_rz

    subroutine backward_rz(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        real*8,intent(in) :: leak
        complex*16,intent(in) :: x(dim_in)
        complex*16,intent(in) :: dy(dim_in)
        complex*16,intent(out) :: dx(dim_in)
        complex*16 :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if(real(xi)<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dy(i)
            endif
            
        enddo
    end subroutine backward_rz
    subroutine forward_riz(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        real*8,intent(in) :: leak
        complex*16,intent(in) :: x(dim_in)
        complex*16,intent(out) :: y(dim_in)
        integer :: i
        complex*16 :: xi
        do i=1,dim_in
            xi=x(i)
            if(aimag(xi)>0 .and. real(xi)>0) then
                y(i)=xi
            else if(aimag(xi)>0 .and. real(xi)<0) then
                y(i)=dcmplx(leak*real(xi),aimag(xi))
            else if(real(xi)<0) then
                y(i)=leak*xi
            else
                y(i)=dcmplx(real(xi),leak*aimag(xi))
            endif
        enddo
    end subroutine forward_riz

    subroutine backward_riz(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        real*8,intent(in) :: leak
        complex*16,intent(in) :: x(dim_in)
        complex*16,intent(in) :: dy(dim_in)
        complex*16,intent(out) :: dx(dim_in)
        complex*16 :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if(aimag(xi)>0 .and. real(xi)>0) then
                dx(i)=dy(i)
            else if(aimag(xi)>0 .and. real(xi)<0) then
                dx(i)=dcmplx(leak*real(dy(i)),aimag(dy(i)))
            else if(real(xi)<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dcmplx(real(dy(i)),leak*aimag(dy(i)))
            endif
        enddo
    end subroutine backward_riz
    
    subroutine forward_rc(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        real*4,intent(in) :: leak
        complex*8,intent(in) :: x(dim_in)
        complex*8,intent(out) :: y(dim_in)
        integer :: i
        complex*8 :: xi
        do i=1,dim_in
            xi=x(i)
            if(real(xi)<0) then
                y(i)=leak*xi
            else
                y(i)=xi
            endif
            
        enddo
    end subroutine forward_rc

    subroutine backward_rc(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        real*4,intent(in) :: leak
        complex*8,intent(in) :: x(dim_in)
        complex*8,intent(in) :: dy(dim_in)
        complex*8,intent(out) :: dx(dim_in)
        complex*8 :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if(real(xi)<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)=dy(i)
            endif
            
        enddo
    end subroutine backward_rc
    subroutine forward_ric(x, y, dim_in, leak)
        implicit none
        integer,intent(in) :: dim_in
        real*4,intent(in) :: leak
        complex*8,intent(in) :: x(dim_in)
        complex*8,intent(out) :: y(dim_in)
        integer :: i
        complex*8 :: xi
        do i=1,dim_in
            xi=x(i)
            if(aimag(xi)>0 .and. real(xi)>0) then
                y(i)=xi
            else if(aimag(xi)>0 .and. real(xi)<0) then
                y(i)=cmplx(leak*real(xi),aimag(xi))
            else if(real(xi)<0) then
                y(i)=leak*xi
            else
                y(i)=cmplx(real(xi),leak*aimag(xi))
            endif
        enddo
    end subroutine forward_ric

    subroutine backward_ric(dy,x,dx,dim_in,leak)
        implicit none
        integer,intent(in) :: dim_in
        real*4,intent(in) :: leak
        complex*8,intent(in) :: x(dim_in)
        complex*8,intent(in) :: dy(dim_in)
        complex*8,intent(out) :: dx(dim_in)
        complex*8 :: xi

        integer :: i

        do i=1,dim_in
            xi=x(i)
            if(aimag(xi)>0 .and. real(xi)>0) then
                dx(i)=dy(i)
            else if(aimag(xi)>0 .and. real(xi)<0) then
                dx(i)=cmplx(leak*real(dy(i)),aimag(dy(i)))
            else if(real(xi)<0) then
                dx(i)=leak*dy(i)
            else
                dx(i)=cmplx(real(dy(i)),leak*aimag(dy(i)))
            endif
        enddo
    end subroutine backward_ric
    
    subroutine forward_rd(x, y, dim_in, leak)
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
    end subroutine forward_rd

    subroutine backward_rd(dy,x,dx,dim_in,leak)
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
    end subroutine backward_rd
    
    subroutine forward_rs(x, y, dim_in, leak)
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
    end subroutine forward_rs

    subroutine backward_rs(dy,x,dx,dim_in,leak)
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
    end subroutine backward_rs
    
    
end module lib