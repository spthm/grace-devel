program ray_test

use ray_mod
use myf03_mod
use mt19937_mod, only: genrand_real3, init_mersenne_twister
use ISO_C_BINDING
implicit none
logical, external :: cpp_src_ray_pluecker
external :: cu_src_ray_pluecker, cu_ray_slope

    integer(C_INT), parameter :: N_cells = 100000, N_rays = 100
    type(src_ray_type) :: src_ray
    type(cpp_src_ray_type) :: cpp_src_ray
    type(slope_src_ray_type) :: slope_src_ray
    integer*4 :: i, j
    logical(C_BOOL) :: hits(N_cells), cpp_hits(N_cells)
    logical(C_BOOL) :: cu_hits(N_cells), slope_hits(N_cells)
    integer(i8b) :: cu_success(8), cu_fail(8)
    integer(i8b) :: slope_success(26), slope_fail(26)
    integer(i8b) :: cu_hit_success(8), cu_miss_success(8)
    integer(i8b) :: cu_hit_fail(8), cu_miss_fail(8)
    real(C_DOUBLE) :: bots(N_cells,3), tops(N_cells,3)
    real(C_DOUBLE) :: s2bs(N_cells,3), s2ts(N_cells,3)
    real(C_DOUBLE) :: tmp(3)
    integer*8 :: seed, N_hits = 0
    character(len=50) :: formatter

    do i=1,8
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

    ! Initialize RNG using count of processor clocks.
    ! Statistically this makes no difference, but it means
    ! the rays and boxes are "always" different.
    call system_clock(seed)
    call init_mersenne_twister(seed)

    ! Loop over different rays.
    do j=1,N_rays
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

        ! Make N_cells cells.  Loop over them in Fortran.
        do i=1,N_cells
            ! Random point (x, y, z) in (-1, 1).
            bots(i,:) = (/ 2.0*genrand_real3()-1.0, &
                           2.0*genrand_real3()-1.0, &
                           2.0*genrand_real3()-1.0 /)

            ! A second random point, such that a line drawn between the two
            ! crosses no axes (its magnitude is greater for each co-ordinate).
            tmp = (/ sign((1.0-abs(bots(i,1)))*genrand_real3() + abs(bots(i,1)), &
                          bots(i,1)), &
                     sign((1.0-abs(bots(i,2)))*genrand_real3() + abs(bots(i,2)), &
                          bots(i,2)), &
                     sign((1.0-abs(bots(i,3)))*genrand_real3() + abs(bots(i,3)), &
                          bots(i,3)) /)

            ! Store first point before finding true bottom/top using min/max.
            tops(i,:) = bots(i,:)

            ! Ensure bots/tops contain the actual minimum and maximum values.
            bots(i,:) = min(bots(i,:), tmp)
            tops(i,:) = max(tops(i,:), tmp)

            ! Ray start -> box bottom/top.
            s2bs(i,:) = bots(i,:) - src_ray%start
            s2ts(i,:) = tops(i,:) - src_ray%start

            ! Check for hit with original Fortran code, increment total hits if so.
            hits(i) = src_ray_pluecker(src_ray, s2bs(i,:), s2ts(i,:))
            if (hits(i) .eqv. .true.) then
                N_hits = N_hits + 1
            endif
        enddo

        ! It is best to pass in pre-allocated cu_hits; if we malloc this in a
        ! C++ function then it may never be deleted!
        ! Check for hit with CUDA code.
        call cu_src_ray_pluecker(cpp_src_ray, s2bs, s2ts, N_cells, cu_hits)

        ! Check for hit with CUDA ray slopes code.
        call cu_ray_slope(slope_src_ray, bots, tops, N_cells, slope_hits)

        ! Loop through hits and check results against Fortran code.
        do i=1,N_cells
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
            if (slope_hits(i) .neqv. hits(i)) then
                slope_fail(slope_src_ray%classification+1) = &
                    slope_fail(slope_src_ray%classification+1) + 1
            else ! slope_hits(i) != hits(i)
                slope_success(slope_src_ray%classification+1) = &
                    slope_success(slope_src_ray%classification+1) + 1
            endif
        enddo
    enddo

    ! Sum hit/miss fails to get total number of incorrect CUDA returns.
    do i=1,8
        cu_success(i) = cu_hit_success(i) + cu_miss_success(i)
        cu_fail(i) = cu_hit_fail(i) + cu_miss_fail(i)
    enddo

    ! Print the total number of correct CUDA results.
    formatter = "(A6, A10, A10, A8)"
    write(*,formatter), "Class ", "CUDA OK", "CUDA Bad", "Ratio"

    formatter = "(I3, A5, I13, A5, I7, A5, F11.2, A4)"
    do i=1,8
        write(*,formatter) i-1, &
                           ""//achar(27)//"[32m", cu_success(i), &
                           ""//achar(27)//"[31m", cu_fail(i), &
                           ""//achar(27)//"[33m", FLOAT(cu_fail(i))/ &
                                                  FLOAT(cu_success(i)), &
                           ""//achar(27)//"[0m"
    enddo

    ! Print the total number of correct CUDA ray-slopes results.
    write(*,*)
    formatter = "(A6, A12, A13, A8)"
    write(*,formatter), "Class ", "Slopes OK", "Slopes Bad", "Ratio"

    formatter = "(I3, A5, I14, A5, I10, A5, F12.2, A4)"
    do i=1,8
        write(*,formatter) i-1, &
                           ""//achar(27)//"[32m", slope_success(i), &
                           ""//achar(27)//"[31m", slope_fail(i), &
                           ""//achar(27)//"[33m", FLOAT(slope_fail(i))/ &
                                                  FLOAT(slope_success(i)), &
                           ""//achar(27)//"[0m"
    enddo

    ! Print the hit/miss-specific numbers of correct CUDA results.
    write(*,*)
    formatter = "(A6, A14, A15, A16, A17)"
    write(*,formatter) "Class ", "Hit CUDA OK", "Hit CUDA Bad", &
                       "Miss CUDA OK", "Miss CUDA Bad"

    formatter = "(I3, A5, I13, A5, I14, A5, I17, A5, I14, A4)"
    do i=1,8
        write(*,formatter) i-1, &
                           ""//achar(27)//"[32m", cu_hit_success(i), &
                           ""//achar(27)//"[31m", cu_hit_fail(i), &
                           ""//achar(27)//"[32m", cu_miss_success(i), &
                           ""//achar(27)//"[31m", cu_miss_fail(i), &
                           ""//achar(27)//"[0m"
    enddo


    write(*,*)
    write(*,"(A5, A8, I10, A4)") ""//achar(27)//"[36m", "Hits:", N_hits, &
                                 ""//achar(27)//"[0m"

end program ray_test
