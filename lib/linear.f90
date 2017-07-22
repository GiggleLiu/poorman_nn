!This is an f90 file automatically generated.
!orders: batch_dim, feature_dim_out/in
module lib
    contains
    subroutine forward_z(x, y, weight, bias, num_batch, nfi, nfo)
        implicit none
        integer,intent(in) :: num_batch, nfi, nfo
        complex*16,intent(in) :: x(num_batch, nfi), weight(nfo, nfi), bias(nfo)
        
        complex*16,intent(out) :: y(num_batch, nfo)
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        integer :: i

        do i=1,nfo
            y(:,i)=bias(i)
        enddo

        call zgemm('N', 'T', num_batch, nfo, nfi, one, x, num_batch,&
            weight, nfo, one, y, num_batch)
    end subroutine forward_z

    subroutine backward_z(dy,x, weight,bias, dx, dweight,dbias,&
            nfi,nfo, num_batch, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch,nfi,nfo
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        complex*16,intent(in) :: x(num_batch, nfi), dy(num_batch, nfo), weight(nfo, nfi), bias(nfo)
        
        complex*16,intent(out) :: dweight(nfo, nfi), dbias(nfo), dx(num_batch, nfi)

        integer :: i
        complex*16,parameter :: one=dcmplx(1D0,0D0)
        complex*16,parameter :: zero=dcmplx(0D0,0D0)

        !f2py intent(out) dx, dweight, dbias

        if(do_wgrad) then
            call zgemm('T', 'N', nfo, nfi, num_batch, one, dy, num_batch,&
                conjg(x), num_batch, zero, dweight, nfo)
        endif
        if(do_xgrad) then
            call zgemm('N', 'N', num_batch, nfi, nfo, one, dy, num_batch,&
                conjg(weight), nfo, zero, dx, num_batch)
        endif
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_z
    subroutine forward_d(x, y, weight, bias, num_batch, nfi, nfo)
        implicit none
        integer,intent(in) :: num_batch, nfi, nfo
        real*8,intent(in) :: x(num_batch, nfi), weight(nfo, nfi), bias(nfo)
        
        real*8,intent(out) :: y(num_batch, nfo)
        real*8,parameter :: one=1D0
        integer :: i

        do i=1,nfo
            y(:,i)=bias(i)
        enddo

        call dgemm('N', 'T', num_batch, nfo, nfi, one, x, num_batch,&
            weight, nfo, one, y, num_batch)
    end subroutine forward_d

    subroutine backward_d(dy,x, weight,bias, dx, dweight,dbias,&
            nfi,nfo, num_batch, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch,nfi,nfo
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*8,intent(in) :: x(num_batch, nfi), dy(num_batch, nfo), weight(nfo, nfi), bias(nfo)
        
        real*8,intent(out) :: dweight(nfo, nfi), dbias(nfo), dx(num_batch, nfi)

        integer :: i
        real*8,parameter :: one=1D0
        real*8,parameter :: zero=0D0

        !f2py intent(out) dx, dweight, dbias

        if(do_wgrad) then
            call dgemm('T', 'N', nfo, nfi, num_batch, one, dy, num_batch,&
                x, num_batch, zero, dweight, nfo)
        endif
        if(do_xgrad) then
            call dgemm('N', 'N', num_batch, nfi, nfo, one, dy, num_batch,&
                weight, nfo, zero, dx, num_batch)
        endif
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_d
    subroutine forward_s(x, y, weight, bias, num_batch, nfi, nfo)
        implicit none
        integer,intent(in) :: num_batch, nfi, nfo
        real*4,intent(in) :: x(num_batch, nfi), weight(nfo, nfi), bias(nfo)
        
        real*4,intent(out) :: y(num_batch, nfo)
        real*4,parameter :: one=1.0
        integer :: i

        do i=1,nfo
            y(:,i)=bias(i)
        enddo

        call sgemm('N', 'T', num_batch, nfo, nfi, one, x, num_batch,&
            weight, nfo, one, y, num_batch)
    end subroutine forward_s

    subroutine backward_s(dy,x, weight,bias, dx, dweight,dbias,&
            nfi,nfo, num_batch, do_xgrad, do_wgrad, do_bgrad)
        implicit none
        integer,intent(in) :: num_batch,nfi,nfo
        logical,intent(in) :: do_xgrad, do_wgrad, do_bgrad
        real*4,intent(in) :: x(num_batch, nfi), dy(num_batch, nfo), weight(nfo, nfi), bias(nfo)
        
        real*4,intent(out) :: dweight(nfo, nfi), dbias(nfo), dx(num_batch, nfi)

        integer :: i
        real*4,parameter :: one=1.0
        real*4,parameter :: zero=0.0

        !f2py intent(out) dx, dweight, dbias

        if(do_wgrad) then
            call sgemm('T', 'N', nfo, nfi, num_batch, one, dy, num_batch,&
                x, num_batch, zero, dweight, nfo)
        endif
        if(do_xgrad) then
            call sgemm('N', 'N', num_batch, nfi, nfo, one, dy, num_batch,&
                weight, nfo, zero, dx, num_batch)
        endif
        if(do_bgrad) then
            !calculate dbias
            dbias=sum(dy,1)
        endif
    end subroutine backward_s
    end module lib