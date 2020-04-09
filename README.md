# beamforming
不同波束形成算法仿真，共计30余种


# 使用方法

在beamforming_compare.py中使用append方法添加一个实例，然后可以查看不同阵列特性，如：

'''Python
    beam.append_beamforming(
    {'beam_domain': 'time',
    'beam_type'  : 'fixed',
    'beam_name'  : 'DelayAndSum'},
    {'d_mic':  2,
    'M': 8,
    'phi':90}
    )
'''

# 项目参考书籍
J.Benesty, I.Cohen, and J.Chen, Fundamentals of Signal Enhancement and Array Signal Processing. Weily-IEEE Press 2017.

# Author 

zhenyuhuang0501@gmail.com