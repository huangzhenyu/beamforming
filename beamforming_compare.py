from beamforming import BeamformingViewer as bv

beam = bv()


beam.append_beamforming(
{'beam_domain': 'time',
 'beam_type'  : 'fixed',
 'beam_name'  : 'DelayAndSum'},
{'d_mic':  2,
 'M': 8,
 'phi':90}
)

# beam.append_beamforming(
# {'beam_domain': 'frequency',
#  'beam_type'  : 'adaptive',
#  'beam_name'  : 'Tradeoff'},
# {'d_mic':1, 'M': 8, 'f0':3400, 
#  'samples':512, 'phi':70, 'amplitude':0.5, 
#  'interferences_var':[1, 1], 'interferences_direction':[30, 90], 
#  'sensor_noise_var': 0.01, 'iSNR_dB': 5, 'miu':0.4}
# )

# beam.append_beamforming(
# {'beam_domain': 'frequency',
#  'beam_type'  : 'adaptive',
#  'beam_name'  : 'Wiener'},
# {'d_mic':1, 'M': 8, 'f0':3400, 
#  'samples':512, 'phi':70, 'amplitude':0.5, 
#  'interferences_var':[1, 1], 'interferences_direction':[30, 90], 
#  'sensor_noise_var': 0.01, 'iSNR_dB': 5}
# )

# beam.append_beamforming(
# {'beam_domain': 'frequency',
#  'beam_type'  : 'adaptive',
#  'beam_name'  : 'MVDR'},
# {'d_mic':1, 'M': 8, 'f0':3400, 
#  'samples':512, 'phi':70, 'amplitude':0.5, 
#  'interferences_var':[1, 1], 'interferences_direction':[30, 90], 
#  'sensor_noise_var': 0.01, 'iSNR_dB': 5}
# )
# beam.append_beamforming(
# {'beam_domain': 'time',
#  'beam_type'  : 'adaptive',
#  'beam_name'  : 'Wiener'},
# {'autocorrelation':0.8, 'phi':0, 'interference_var':1,
# 'interference_direction':90, 'sensor_noise_var':0.1, 'iSNR_dB':5, 
# 'd_mic':3, 'M':30, 'time_samples':30, 'sinc_samples':20, 'fs':8000}
# )

# beam.append_beamforming(
# {'beam_domain': 'frequency',
#  'beam_type'  : 'pattern_design',
#  'beam_name'  : 'FreqInvariant'},
# {'d_mic': 0.5,
#  'M':4, 
#  'order':1, 
#  'epsilon':2,
#  'alpha':[0, 1]}
# )

# beam.append_beamforming(
# {'beam_domain': 'frequency',
#  'beam_type'  : 'differential',
#  'beam_name'  : 'FirstOrder'},
# {'d_mic':  2,
#  'pattern':'Supercardioid'}
# )

# beam.append_beamforming(
# {'beam_domain': 'frequency',
#  'beam_type'  : 'fixed',
#  'beam_name'  : 'DelayAndSum'},
# {'d_mic':  2,
#  'M': 8,
#  'phi': 0}
# )

########################################################
# 1.数beam_pattern 用来查看所有beamforming实例的空间特性，可选参数为
#  polor选择坐标系是为极坐标还是直角坐标，save_fig用来保存图片   
beam.beam_pattern(polar=False, save_fig=True)

#########################################################
# 2.函数 heatmap_beam_pattern 用来查看模式的热力图，频率，角度,save_fig用来保存图片   
# beam.heatmap_beam_pattern() 


###########FixedBeamforming 特有##########################
# 3.函数 white_noise_gain 和directivity 用来查看白噪声增益和指向性因数  
# beam.white_noise_gain(False)
# beam.directivity(False)


###########AdaptiveBeamforming 特有##########################
# beam.noise_reduction_signal_distortion_factor()