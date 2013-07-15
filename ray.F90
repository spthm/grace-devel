module ray_mod

use myf03_mod
use mt19937_mod, only: genrand_real1
! For converting derived types to structs
use ISO_C_BINDING
implicit none

private

public :: src_ray_type
public :: src_ray_make

public :: src_ray_class_from_dir
public :: src_ray_pluecker

public :: cpp_src_ray_type
public :: slope_src_ray_type

real(r8b), parameter :: zero = 0.0d0
real(r8b), parameter :: one = 1.0d0


!> basic ray type
!-----------------------------------------------------------------------
type, BIND(C) :: src_ray_type
  real(C_DOUBLE) :: start(3)  !< starting position
  real(C_DOUBLE) :: dir(3)    !< unit vector direction
  real(C_DOUBLE) :: length    !< length (determines when to stop tracing)
  integer(C_INT) :: class  !< based on direction signs (MMM, PMM, ...)

  real(C_DOUBLE) :: freq      !< freq in HI ionizing units
  real(C_DOUBLE) :: enrg      !< enrg of a single photon in ergs
  real(C_DOUBLE) :: pcnt      !< photon count (changes as the ray is depleted)
  real(C_DOUBLE) :: pini      !< initial photons
  real(C_DOUBLE) :: dt_s      !< time step associated with ray [s]
end type src_ray_type

type, BIND(C) :: cpp_src_ray_type
  real(C_DOUBLE) :: start(3)  !< starting position
  real(C_DOUBLE) :: dir(3)    !< unit vector direction
  real(C_DOUBLE) :: length    !< length (determines when to stop tracing)
  integer(C_INT) :: dir_class  !< based on direction signs (MMM, PMM, ...)

  real(C_DOUBLE) :: freq      !< freq in HI ionizing units
  real(C_DOUBLE) :: enrg      !< enrg of a single photon in ergs
  real(C_DOUBLE) :: pcnt      !< photon count (changes as the ray is depleted)
  real(C_DOUBLE) :: pini      !< initial photons
  real(C_DOUBLE) :: dt_s      !< time step associated with ray [s]
end type cpp_src_ray_type

type, BIND(C) :: slope_src_ray_type
  integer(C_INT) :: classification !< The position of this matters for some reason.
  real(C_DOUBLE) :: x, y, z  !< starting position
  real(C_DOUBLE) :: i, j, k    !< unit vector direction
  real(C_FLOAT)  :: ibyj, jbyi, kbyj, jbyk, ibyk, kbyi !< Slope
  real(C_FLOAT)  :: c_xy, c_xz, c_yx, c_yz, c_zx, c_zy
  real(C_DOUBLE) :: length    !< length (determines when to stop tracing)

  real(C_DOUBLE) :: freq      !< freq in HI ionizing units
  real(C_DOUBLE) :: enrg      !< enrg of a single photon in ergs
  real(C_DOUBLE) :: pcnt      !< photon count (changes as the ray is depleted)
  real(C_DOUBLE) :: pini      !< initial photons
  real(C_DOUBLE) :: dt_s      !< time step associated with ray [s]
end type slope_src_ray_type



contains

!> creates a source ray
!-----------------------------------------------------------------------
  subroutine src_ray_make(ray)

    type(src_ray_type), intent(out) :: ray    !< ray to make
    real(r8b) :: xx,yy,zz,r

    ! random direction on the unit sphere
    !-----------------------------------------------------------------------
    r=2.0d0
    do while ( r .GT. 1.0d0 .and. r .NE. 0.0d0 )
       xx=(2.0d0 * genrand_real1()-1.0d0)
       yy=(2.0d0 * genrand_real1()-1.0d0)
       zz=(2.0d0 * genrand_real1()-1.0d0)
       r=xx*xx+yy*yy+zz*zz
    enddo
    r = sqrt(r)
    ray%dir(1) = xx/r  ! it is important that ray%dir be a unit vector
    ray%dir(2) = yy/r
    ray%dir(3) = zz/r

    !-----------------------------------------------------------------------

    ray%start = (/ zero, zero, zero /)

    ray%length = huge(1.0d0) * 0.1d0

!   set the class of the ray (what octant is it going into)
    call src_ray_class_from_dir(ray)

  end subroutine src_ray_make


! pre computes the class of the ray for the Pluecker test
! ray label    class
!   MMM          0
!   PMM          1
!   MPM          2
!   PPM          3
!   MMP          4
!   PMP          5
!   MPP          6
!   PPP          7
!-----------------------------------------------------------
subroutine src_ray_class_from_dir( src_ray )
  type(src_ray_type), intent(inout) :: src_ray
  integer(i4b) :: i

  src_ray%class = 0
  do i = 1, 3
     if ( src_ray%dir(i) >= zero ) src_ray%class = src_ray%class + 2**(i-1)
  end do

end subroutine src_ray_class_from_dir


!> pluecker test for line segment / cell intersection
!-----------------------------------------------------
function src_ray_pluecker(src_ray, s2b, s2t) result( hit )

  type(src_ray_type), intent(in) :: src_ray

  real(r8b) :: s2b(3)         !< vector from ray start to lower cell corner
  real(r8b) :: s2t(3)         !< vector from ray start to upper cell corner
  logical :: hit              !< true or false result

  real(r8b) :: dir(3)
  real(r8b) :: dist

  real(r8b) :: e2b(3)       !< vector from ray end to lower cell corner
  real(r8b) :: e2t(3)       !< vector from ray end to upper cell corner

  dir  = src_ray%dir
  dist = src_ray%length

  e2b = s2b - dir * dist
  e2t = s2t - dir * dist

  hit = .false.

  ! branch on ray direction
  !---------------------------
  select case( src_ray%class )

     ! MMM
     !-----------
  case(0)

     if(s2b(1) > zero .or. s2b(2) > zero .or. s2b(3) > zero) return ! on negative part of ray
     if(e2t(1) < zero .or. e2t(2) < zero .or. e2t(3) < zero) return ! past length of ray

     if ( dir(1)*s2b(2) - dir(2)*s2t(1) < zero .or.  &
          dir(1)*s2t(2) - dir(2)*s2b(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) > zero       ) return

     ! PMM
     !-----------
  case(1)

     if(s2t(1) < zero .or. s2b(2) > zero .or. s2b(3) > zero) return ! on negative part of ray
     if(e2b(1) > zero .or. e2t(2) < zero .or. e2t(3) < zero) return ! past length of ray

     if ( dir(1)*s2t(2) - dir(2)*s2t(1) < zero .or.  &
          dir(1)*s2b(2) - dir(2)*s2b(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) > zero       ) return

     ! MPM
     !-----------
  case(2)

     if(s2b(1) > zero .or. s2t(2) < zero .or. s2b(3) > zero) return ! on negative part of ray
     if(e2t(1) < zero .or. e2b(2) > zero .or. e2t(3) < zero) return ! past length of ray

     if ( dir(1)*s2b(2) - dir(2)*s2b(1) < zero .or.  &
          dir(1)*s2t(2) - dir(2)*s2t(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) > zero       ) return

     ! PPM
     !-----------
  case(3)

     if(s2t(1) < zero .or. s2t(2) < zero .or. s2b(3) > zero) return ! on negative part of ray
     if(e2b(1) > zero .or. e2b(2) > zero .or. e2t(3) < zero) return ! past length of ray

     if ( dir(1)*s2t(2) - dir(2)*s2b(1) < zero .or.  &
          dir(1)*s2b(2) - dir(2)*s2t(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) > zero       ) return

     ! MMP
     !-----------
  case(4)

     if(s2b(1) > zero .or. s2b(2) > zero .or. s2t(3) < zero) return ! on negative part of ray
     if(e2t(1) < zero .or. e2t(2) < zero .or. e2b(3) > zero) return ! past length of ray

     if ( dir(1)*s2b(2) - dir(2)*s2t(1) < zero .or.  &
          dir(1)*s2t(2) - dir(2)*s2b(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) > zero       ) return


     ! PMP
     !-----------
  case(5)

     if(s2t(1) < zero .or. s2b(2) > zero .or. s2t(3) < zero) return ! on negative part of ray
     if(e2b(1) > zero .or. e2t(2) < zero .or. e2b(3) > zero) return ! past length of ray

     if ( dir(1)*s2t(2) - dir(2)*s2t(1) < zero .or.  &
          dir(1)*s2b(2) - dir(2)*s2b(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2b(2) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2t(2) > zero       ) return


     ! MPP
     !-----------
  case(6)

     if(s2b(1) > zero .or. s2t(2) < zero .or. s2t(3) < zero) return ! on negative part of ray
     if(e2t(1) < zero .or. e2b(2) > zero .or. e2b(3) > zero) return ! past length of ray

     if ( dir(1)*s2b(2) - dir(2)*s2b(1) < zero .or.  &
          dir(1)*s2t(2) - dir(2)*s2t(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2t(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2b(1) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) > zero       ) return

     ! PPP
     !-----------
  case(7)

     if(s2t(1) < zero .or. s2t(2) < zero .or. s2t(3) < zero) return ! on negative part of ray
     if(e2b(1) > zero .or. e2b(2) > zero .or. e2b(3) > zero) return ! past length of ray

     if ( dir(1)*s2t(2) - dir(2)*s2b(1) < zero .or.  &
          dir(1)*s2b(2) - dir(2)*s2t(1) > zero .or.  &
          dir(1)*s2b(3) - dir(3)*s2t(1) > zero .or.  &
          dir(1)*s2t(3) - dir(3)*s2b(1) < zero .or.  &
          dir(2)*s2t(3) - dir(3)*s2b(2) < zero .or.  &
          dir(2)*s2b(3) - dir(3)*s2t(2) > zero       ) return

  case default
     call rayError('ray class.')

  end select

  hit=.true.

end function src_ray_pluecker


!> error handling
!-----------------------------
  subroutine rayError(string,i)
    character(*) :: string  !< error string
    integer, optional :: i  !< error number

    print*,' Error detected:'

    if(present(i)) then
       print*,string,i
    else
       print*,string
    endif

    stop
  end subroutine rayError


end module ray_mod
