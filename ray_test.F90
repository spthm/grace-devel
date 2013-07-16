program ray_test

use ray_mod
use myf03_mod
use mt19937_mod, only: genrand_real3, init_mersenne_twister
use ISO_C_BINDING
implicit none
external :: cu_src_ray_pluecker, cu_ray_slope

    integer(C_INT), parameter :: N_cells = 10000, N_rays = 100
    type(src_ray_type) :: src_ray
    type(cpp_src_ray_type) :: cpp_src_ray
    integer*4 :: i, j, k
    logical(C_BOOL) :: hits(N_cells)
    logical(C_BOOL) :: cu_hits(N_cells)
    integer(i8b) :: cu_success(8), cu_fail(8)
    integer(i8b) :: cu_hit_success(8), cu_miss_success(8)
    integer(i8b) :: cu_hit_fail(8), cu_miss_fail(8)
    real(C_FLOAT) :: cu_time = 0, fortran_time = 0, elapsed_time
    integer(C_LONG) :: init_time, final_time, count_rate
    real(C_DOUBLE) :: bots(N_cells,3), tops(N_cells,3)
    real(C_DOUBLE) :: s2b(3), s2t(3)
    real(C_DOUBLE) :: mag, tmp(3)
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

        ! Make N_cells cells.  Loop over them in Fortran.
        do i=1,N_cells
            ! Random point (x, y, z) in (-1, 1).
            do k=1,3
                bots(i,k) = 2.0*genrand_real3()-1.0
            enddo

            ! A second random point, such that a line drawn between the two
            ! crosses no axes (its magnitude is greater for each co-ordinate).
            do k=1,3
                mag = abs(bots(i,k))
                tmp(k) = sign( (1.0-mag)*genrand_real3() + mag, bots(i,k) )
            enddo

            ! Store first point before finding true bottom/top using min/max.
            tops(i,:) = bots(i,:)

            ! Ensure bots/tops contain the actual minimum and maximum values.
            bots(i,:) = min(bots(i,:), tmp)
            tops(i,:) = max(tops(i,:), tmp)

            ! Ray start -> box bottom/top.
            s2b = bots(i,:) - src_ray%start
            s2t = tops(i,:) - src_ray%start

            ! Check for hit with original Fortran code, increment total hits if so.
            call system_clock(init_time, count_rate)

            hits(i) = src_ray_pluecker(src_ray, s2b, s2t)

            call system_clock(final_time)
            fortran_time = fortran_time + (final_time - init_time)

            if (hits(i) .eqv. .true.) then
                N_hits = N_hits + 1
            endif
        enddo

        ! It is best to pass in pre-allocated cu_hits; if we malloc this in a
        ! C++ function then it may never be deleted!
        ! Check for hit with CUDA code.
        call cu_src_ray_pluecker(cpp_src_ray, bots, tops, N_cells, &
                                 cu_hits, elapsed_time)
        cu_time = cu_time + elapsed_time

        ! Loop through hits and check results against Fortran code.
        do i=1,N_cells
             ! Handle case that CUDA result != Fortran result.
            if (hits(i) .eqv. .true.) then
                if (cu_hits(i) .neqv. hits(i)) then
                    cu_hit_fail(src_ray%class+1) = &
                        cu_hit_fail(src_ray%class+1) + 1
                else ! cu_hits(i) == hits(i)
                    cu_hit_success(src_ray%class+1) = &
                        cu_hit_success(src_ray%class+1) + 1
                endif
            else ! hits(i) == false
                if (cu_hits(i) .neqv. hits(i)) then
                    cu_miss_fail(src_ray%class+1) = &
                        cu_miss_fail(src_ray%class+1) + 1
                else ! cu_hits(i) == hits(i)
                    cu_miss_success(src_ray%class+1) = &
                        cu_miss_success(src_ray%class+1) + 1
                endif
            endif
        enddo
    enddo

    ! Convert Fortran code time to milliseconds.
    fortran_time = 1000.0 * fortran_time / dble(count_rate)

    ! Sum hit/miss fails to get total number of incorrect CUDA returns.
    do i=1,8
        cu_success(i) = cu_hit_success(i) + cu_miss_success(i)
        cu_fail(i) = cu_hit_fail(i) + cu_miss_fail(i)
    enddo

    ! Print the total number of correct CUDA results.
    formatter = "(A6, A10, A10, A8)"
    write(*,formatter), "Class ", "CUDA OK", "CUDA Err", "Ratio"

    formatter = "(I3, A5, I13, A5, I7, A5, F11.2, A4)"
    do i=1,8
        write(*,formatter) i-1, &
                           ""//achar(27)//"[32m", cu_success(i), &
                           ""//achar(27)//"[31m", cu_fail(i), &
                           ""//achar(27)//"[33m", FLOAT(cu_fail(i))/ &
                                                  FLOAT(cu_success(i)), &
                           ""//achar(27)//"[0m"
    enddo

    ! Print the hit/miss-specific numbers of correct CUDA results.
    write(*,*)
    formatter = "(A6, A14, A15, A16, A17)"
    write(*,formatter) "Class ", "Hit CUDA OK", "Hit CUDA Err", &
                       "Miss CUDA OK", "Miss CUDA Err"

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

    write(*,*)
    write(*,"(A19, F8.3, A3)") "Total Fortran time: ", fortran_time, " ms"
    write(*,"(A16, F8.3, A3)") "Total GPU time: ", cu_time, " ms"

end program ray_test
