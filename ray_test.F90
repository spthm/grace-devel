program ray_test

use ray_mod
use myf03_mod
use mt19937_mod, only: genrand_real3, init_mersenne_twister
use ISO_C_BINDING
implicit none
logical, external :: cpp_src_ray_pluecker, cu_ray_slope
external :: cu_src_ray_pluecker

    integer(C_INT), parameter :: N_cells = 1000
    type(src_ray_type) :: src_ray
    type(cpp_src_ray_type) :: cpp_src_ray
    type(slope_src_ray_type) :: slope_src_ray
    integer(i8b) :: i
    logical(C_BOOL) :: hits(N_cells), cpp_hits(N_cells)
    logical(C_BOOL) :: cu_hits(N_cells), slope_hits(N_cells)
    integer(i8b) :: cpp_success(8), cpp_fail(8)
    integer(i8b) :: cu_success(8), cu_fail(8)
    integer(i8b) :: slope_success(26), slope_fail(26)
    integer(i8b) :: cu_hit_success(8), cu_miss_success(8)
    integer(i8b) :: cu_hit_fail(8), cu_miss_fail(8)
    real(C_DOUBLE) :: bots(N_cells,3), tops(N_cells,3)
    real(C_DOUBLE) :: s2bs(N_cells,3), s2ts(N_cells,3)
    integer*8 :: seed, N_hits = 0
    character(len=50) :: formatter

    do i=1,8
        cpp_success(i) = 0.0
        cpp_fail(i) = 0.0
        cu_success(i) = 0.0
        cu_fail(i) = 0.0
        cu_hit_success(i) = 0.0
        cu_hit_fail(i) = 0.0
        cu_miss_success(i) = 0.0
        cu_miss_fail(i) = 0.0
    enddo

    do i=1,26
        slope_success(i) = 0.0
        slope_fail(i) = 0.0
    enddo

    ! Initialize RNG using current time of day in seconds.
    ! Statistically speaking, this makes no difference, but it means
    ! the rays and boxes are "always" different.
    call system_clock(seed)
    call init_mersenne_twister(seed)

    ! Make the ray.
    call src_ray_make(src_ray)
    ! To test the pluecker ray-length criterion
    !src_ray%length = 0.0001

    ! Copy src_ray into C++ friendly format.
    cpp_src_ray%start = src_ray%start
    cpp_src_ray%dir = src_ray%dir
    cpp_src_ray%length = src_ray%length
    cpp_src_ray%dir_class = src_ray%class

    call make_slope_ray(cpp_src_ray, slope_src_ray)

    ! Make N_cells cells.
    do i=1,N_cells
!         ! Top of cell in (0.75, 1)
        tops(i,:) = (/ 0.25d0*genrand_real3()+0.75d0, &
                       0.25d0*genrand_real3()+0.75d0, &
                       0.25d0*genrand_real3()+0.75d0 /)
        ! Bottom of cell in (0.5, 0.75)
        bots(i,:) = (/ 0.25d0*genrand_real3()+0.5d0, &
                       0.25d0*genrand_real3()+0.5d0, &
                       0.25d0*genrand_real3()+0.5d0 /)

!         tops(i,:) = (/ 1.0, 1.0, 1.0 /)
!         bots(i,:) = (/ -0.5, -0.5, -0.5 /)

        s2bs(i,:) = bots(i,:) - src_ray%start
        s2ts(i,:) = tops(i,:) - src_ray%start

        ! Check for hit with original Fortran code.
        hits(i) = src_ray_pluecker(src_ray, s2bs(i,:), s2ts(i,:))
        ! Increment total hits.
        if (hits(i) .eqv. .true.) then
            N_hits = N_hits + 1
        endif
    enddo

    ! Check for hit with C++ code.
    !cpp_hits = cpp_src_ray_pluecker(cpp_src_ray, s2b, s2t)

    ! Check for hit with CUDA code.
    call cu_src_ray_pluecker(cpp_src_ray, s2bs, s2ts, N_cells, cu_hits)

    ! Check for hit with CUDA ray slopes code.
    !slope_hits = cu_ray_slope(slope_src_ray, bots, tops, N)

    ! Loop through hits and check results against Fortran code.
    do i=1,N_cells
!         ! Handle case that C++ result != Fortran result.
!         if (cpp_hit .neqv. hit) then
!             cpp_fail(src_ray%class+1) = cpp_fail(src_ray%class+1) + 1
!         else
!             cpp_success(src_ray%class+1) = cpp_success(src_ray%class+1) + 1
!         endif

         ! Handle case that CUDA result != Fortran result.
        if (hits(i) .eqv. .true.) then
            if (cu_hits(i) .neqv. hits(i)) then
                cu_hit_fail(src_ray%class+1) = cu_hit_fail(src_ray%class+1) + 1
            else ! cu_hits(i) == hits(i)
                cu_hit_success(src_ray%class+1) = cu_hit_success(src_ray%class+1) + 1
            endif
        else ! hits(i) == false
            if (cu_hits(i) .neqv. hits(i)) then
                cu_miss_fail(src_ray%class+1) = cu_miss_fail(src_ray%class+1) + 1
            else ! cu_hits(i) == hits(i)
                cu_miss_success(src_ray%class+1) = cu_miss_success(src_ray%class+1) + 1
            endif
        endif

        ! Handle case that CUDA slopes-test result != Fortran result.
!         if (slope_hit .neqv. hit) then
!             slope_fail(slope_src_ray%classification+1) = &
!                 slope_fail(slope_src_ray%classification+1) + 1
!         else
!             slope_success(slope_src_ray%classification+1) = &
!                 slope_success(slope_src_ray%classification+1) + 1
!         endif
    enddo

    ! Sum hit/miss fails to get total number of incorrect CUDA returns.
    do i=1,8
        cu_success(i) = cu_hit_success(i) + cu_miss_success(i)
        cu_fail(i) = cu_hit_fail(i) + cu_miss_fail(i)
    enddo

    ! Print the total number of correct C++ and CUDA results.
!     formatter = "(A6, A10, A10, A8, A14, A10, A8)"
!     write(*,formatter), "Class ", "C++ OK", "C++ Bad", "Ratio", &
!                         "CUDA OK", "CUDA Bad", "Ratio"

!     formatter = "(I3, I12, I8, F11.2, I12, I8, F11.2)"
!     do i=1,8
!         write(*,formatter) i-1, cpp_success(i), cpp_fail(i), &
!             FLOAT(cpp_fail(i))/FLOAT(cpp_success(i)), &
!             cu_success(i), cu_fail(i), FLOAT(cu_fail(i))/FLOAT(cu_success(i))
!     enddo

    ! Print the hit/miss-specific numbers of correct CUDA results.
    write(*,*)
    formatter = "(A6, A14, A15, A16, A17)"
    write(*,formatter) "Class ", "CUDA Hit OK", "CUDA Hit Bad", &
                       "CUDA Miss OK", "CUDA Miss Bad"

    formatter = "(I3, I13, I14, I17, I14)"
    do i=1,8
        write(*,formatter) i-1, cu_hit_success(i), cu_hit_fail(i), &
                           cu_miss_success(i), cu_miss_fail(i)
    enddo

    ! Print the total number of correct CUDA ray-slopes results.
!     write(*,*)
!     formatter = "(A6, A12, A12, A8)"
!     write(*,formatter), "Class ", "Slopes OK", "Slopes Bad", "Ratio"

!     formatter = "(I3, I14, I10, F11.2)"
!     do i=1,26
!         write(*,formatter) i-1, slope_success(i), slope_fail(i), &
!             FLOAT(slope_fail(i))/FLOAT(slope_success(i))
!     enddo

    write(*,*)
    write(*,"(A8, I6)") "Hits:", N_hits

end program ray_test
