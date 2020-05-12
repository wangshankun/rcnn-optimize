#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math

set_len  = 64
way_len  = 8
line_len = 64
int_len  = 4.0

#第一列举列子查看方阵每个set组命被使用次数
def cache_set_used_num(square_matrix_row):
    sta_dict = {}
    for i in range(square_matrix_row):
        set_num = int(math.ceil(((square_matrix_row * i + 0)* int_len )/line_len))%set_len
        if sta_dict.has_key(set_num):
            sta_dict[set_num] = sta_dict[set_num]+1;
        else:
            sta_dict[set_num] = 1;
    return sta_dict
    

print("square_matrix_row 512 the 0 col cache_set_used_num:")
print(cache_set_used_num(512))
print("square_matrix_row 513 the 0 col cache_set_used_num:")
print(cache_set_used_num(513))
