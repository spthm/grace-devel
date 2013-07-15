program ray_test

use ray_float_mod
use myf03_mod
use mt19937_mod, only: genrand_real3, init_mersenne_twister
implicit none
logical, external :: cpp_src_ray_pluecker_float, cu_src_ray_pluecker_float

    type(src_ray_type) :: src_ray
    type(cpp_src_ray_type) :: cpp_src_ray
    integer(i8b) :: i
    integer(i8b), parameter :: N = 10000
    logical :: hit, cpp_hit, cu_hit
    integer(i8b) :: cpp_success(8), cpp_fail(8)
    integer(i8b) :: cu_success(8), cu_fail(8)
    integer(i8b) :: cu_hit_success(8), cu_miss_success(8)
    integer(i8b) :: cu_hit_fail(8), cu_miss_fail(8)
    real(r4b) :: botrange(3), bot(3)
    real(r4b) :: toprange(3), top(3)
    integer*4 :: time(3)
    integer*8 :: time_s, hits = 0
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

    ! Initialize RNG using current time of day in seconds.
    ! Statistically speaking, this makes no difference, but it means
    ! the rays and boxes are "always" different.
    call itime(time)
    time_s = time(3) + 60*(time(2) + 60*time(1))
    call init_mersenne_twister(time_s)

    do i=1,N
        call src_ray_make(src_ray)

        ! To test the ray-length criterion
        !src_ray%length = 0.0001

        ! Copy src_ray into C++ - friendly format.
        cpp_src_ray%start = src_ray%start
        cpp_src_ray%dir = src_ray%dir
        cpp_src_ray%length = src_ray%length
        cpp_src_ray%dir_class = src_ray%class

        ! Top of cell in (0.75, 1)
        toprange = sngl((/ 0.25d0*genrand_real3()+0.75d0, &
                      0.25d0*genrand_real3()+0.75d0, &
                      0.25d0*genrand_real3()+0.75d0 /))
        ! Bottom of cell in (0.5, 0.75)
        botrange = sngl((/ 0.25d0*genrand_real3()+0.5d0, &
                      0.25d0*genrand_real3()+0.5d0, &
                      0.25d0*genrand_real3()+0.5d0 /))

        bot = botrange - src_ray%start
        top = toprange - src_ray%start


        ! Check for hit with original Fortran code.
        hit = src_ray_pluecker_float(src_ray, bot, top)

        ! Check for hit with C++ code.
        cpp_hit = cpp_src_ray_pluecker_float(cpp_src_ray, bot, top)

        ! Check for hit with CUDA code.
        cu_hit = cu_src_ray_pluecker_float(cpp_src_ray, bot, top)

        ! Increment total hits.
        if (hit .eqv. .true.) then
            hits = hits + 1
        endif

        ! Handle case that C++ result != Fortran result.
        if (cpp_hit .neqv. hit) then
            cpp_fail(src_ray%class+1) = cpp_fail(src_ray%class+1) + 1
        else
            cpp_success(src_ray%class+1) = cpp_success(src_ray%class+1) + 1
        endif

         ! Handle case that CUDA result != Fortran result.
        if (hit .eqv. .true.) then
            if (cu_hit .neqv. hit) then
                cu_hit_fail(src_ray%class+1) = cu_hit_fail(src_ray%class+1) + 1
            else ! cu_hit == hit
                cu_hit_success(src_ray%class+1) = cu_hit_success(src_ray%class+1) + 1
            endif
        else ! hit == false
            if (cu_hit .neqv. hit) then
                cu_miss_fail(src_ray%class+1) = cu_miss_fail(src_ray%class+1) + 1
            else ! cut_hit == hit
                cu_miss_success(src_ray%class+1) = cu_miss_success(src_ray%class+1) + 1
            endif
        endif
    enddo

    ! Sum hit/miss fails to get total number of incorrect CUDA returns.
    do i=1,8
        cu_success(i) = cu_hit_success(i) + cu_miss_success(i)
        cu_fail(i) = cu_hit_fail(i) + cu_miss_fail(i)
    enddo

    ! Print the total number of correct C++ and CUDA results.
    formatter = "(A6, A10, A10, A8, A14, A10, A8)"
    write(*,formatter), "Class ", "C++ OK", "C++ Bad", "Ratio", &
                        "CUDA OK", "CUDA Bad", "Ratio"

    formatter = "(I3, I12, I8, F11.2, I12, I8, F11.2)"
    do i=1,8
        write(*,formatter) i-1, cpp_success(i), cpp_fail(i), &
            FLOAT(cpp_fail(i))/FLOAT(cpp_success(i)), &
            cu_success(i), cu_fail(i), FLOAT(cu_fail(i))/FLOAT(cu_success(i))
    enddo

    ! Print the hit/miss-specific numbers of correct CUDA results.
    write(*,*)
    write(*,*)
    formatter = "(A6, A14, A15, A16, A17)"
    write(*,formatter) "Class ", "CUDA Hit OK", "CUDA Hit Bad", &
                       "CUDA Miss OK", "CUDA Miss Bad"

    formatter = "(I3, I13, I14, I17, I14)"
    do i=1,8
        write(*,formatter) i-1, cu_hit_success(i), cu_hit_fail(i), &
                           cu_miss_success(i), cu_miss_fail(i)
    enddo

    write(*,*)
    write(*,"(A8, I6)") "Hits:", hits

end program ray_test
