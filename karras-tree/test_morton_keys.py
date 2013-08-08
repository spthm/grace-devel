import numpy as np

import morton_keys
import bits

# x = 00110101
# y = 10101110
# => spaced_x =  000010100010001
# => spaced_y = 100010001010100
# => key =      1000110110111001

x = int('0b00110101', 2)
y = int('0b10101110', 2)

true_spaced_x = int('0b0000010100010001', 2)
true_spaced_y = int('0b0100010001010100', 2)
true_key = int('0b1000110110111001', 2)

my_spaced_x = bits.space_by_1(x)
my_spaced_y = bits.space_by_1(y)

my_key = (my_spaced_y << 1) | my_spaced_x
my_key_2 = morton_keys.morton_key_2D(x, y)

print "True 2D key:                        ", true_key
print "Key using bits.space_by_1:          ", my_key
print "Key using morton_keys.morton_key_2D:", my_key_2
print


# x = 00110101
# y = 10101110
# z = 01101011
# => spaced_x =   0000001001000001000001
# => spaced_y =  1000001000001001001000
# => spaced_z = 0001001000001000001001
# => key =      010100111001110011110101

z = int('0b01101011', 2)

true_spaced_x = int('0b0000001001000001000001', 2)
true_spaced_y = int('0b1000001000001001001000', 2)
true_spaced_z = int('0b0001001000001000001001', 2)
true_key = int('0b010100111001110011110101', 2)

my_spaced_x = bits.space_by_2(x)
my_spaced_y = bits.space_by_2(y)
my_spaced_z = bits.space_by_2(z)

my_key = (my_spaced_z << 2) | (my_spaced_y << 1) | my_spaced_x
my_key_2 = morton_keys.morton_key_3D(x, y, z)

print "True 3D key:                        ", true_key
print "Key using bits.space_by_2:          ", my_key
print "Key using morton_keys.morton_key_3D:", my_key_2
