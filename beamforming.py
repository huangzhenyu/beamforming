#@version1.0 date: 11/19/2019 by zhenyuhuang
#@version2.0 daye: 02/20/2020 by zhenyuhuang
#@author:          zhenyuhuang0501@gmail.com
'''
class for comparing the performance of different beamforming.
'''
import os
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from parameters import constants
from util import import_beamforming, make_directory_if_not_exists
from plot import Plot


class BeamformingViewer:
    '''
    compare different performance for one or several beamforming
    such as beampattern(polar or cartesian)/directivity/white_noise_gain
    '''
    def __init__(self):
        self._beamforming_list = []
        self._beamforming_cout = 0     

    def fig_name(self, fig_type):
        '''
        fig name : fig_type combine all the beamformings' label

        Parameter
        ---------
        fig_type: string
            the fig type, such as beampattern polar or cartesian
        '''
        title = ''
        # combine all beamforming instance label together 
        for beam in self._beamforming_list:
            title = '{}_{}'.format(title, beam['beam_instance'].beam_label)
        title = '{}{}'.format(fig_type, title)
        
        pic_path = os.path.join(constants.get('pic_path'), 'beamforming_compare')
        make_directory_if_not_exists(pic_path)

        return os.path.join(pic_path ,'{}.png'.format(title))

    def append_beamforming(self, beamforming_dict, args):

        if self._beamforming_cout == 0:
            if all(key in beamforming_dict for key in ('beam_domain', 'beam_type', 'beam_name')): 
                beamforming = import_beamforming(**beamforming_dict)  
            else: 
                raise ValueError('Beamforming must be specified domain/type/name ' +
                                'at the same time when append first beamforming! Please check it!')
        else:

            # if not one beamforming, the function is comparing. It must
            # keep the same domain and type. 
            beam_prev = self._beamforming_list[-1]
            if 'beam_domain' in beamforming_dict:
                assert beam_prev['beam_domain'] == beamforming_dict['beam_domain']
            else:
                beamforming_dict['beam_domain'] = beam_prev['beam_domain']
            
            if 'beam_type' in beamforming_dict:
                assert beam_prev['beam_type'] == beamforming_dict['beam_type']
            else:
                beamforming_dict['beam_type'] = beam_prev['beam_type']

            if 'beam_name' not in beamforming_dict:
                raise ValueError('Beamforming must have a beam name!')
            
            beamforming = import_beamforming(**beamforming_dict)

        
        # if beam_type is adaptive, must initialize a soundfield first.
        if beamforming_dict['beam_type'] == 'adaptive':
            if beamforming_dict['beam_domain'] == 'time':
                from time_domain_beamforming.sound_field import SoundField
                sound_field_para = ['autocorrelation', 'phi', 'interference_var',
                            'interference_direction', 'sensor_noise_var', 'iSNR_dB', 
                            'd_mic', 'M', 'time_samples', 'sinc_samples', 'fs']

            elif beamforming_dict['beam_domain'] == 'frequency':
                from frequency_domain_beamforming.sound_field import SoundField
                sound_field_para = ['f0', 'samples', 'phi', 'amplitude', 
                                    'interferences_var', 'interferences_direction',
                                     'sensor_noise_var', 'iSNR_dB', 'd_mic', 'M']

            for para in sound_field_para:
                if para not in args.keys():
                    raise ValueError('Adaptive Beamforming need {} paras to construc a sound field!'.format(para))
            
            def extract_dict_by_key(ini_dict, keylist, return_remain=True):
                dict_key1 = {}
                dict_key2 = {}
                for key, value in ini_dict.items():
                    if key in keylist:
                        dict_key1.update({key:value})
                    else:
                        dict_key2.update({key:value})
                if return_remain:
                    return dict_key1, dict_key2
                else:
                    return dict_key1   

            sd = SoundField()
            args1, args2 = extract_dict_by_key(args, sound_field_para)
            sd.build_sound_field(**args1)
            beam = {'beam_instance' : beamforming(sd.sound_field, **args2)}
            beam.update(beamforming_dict)
            beam.update(args)


        else:
            assert 'd_mic'       in args
            # assert 'M'           in args
            beam = {'beam_instance' : beamforming(**args)}
            beam.update(beamforming_dict)
            beam.update(args)

        self._beamforming_list.append(beam)
        self._beamforming_cout += 1


    def beam_pattern(self, polar, save_fig):
        '''
        plot either beamforming or beamforming group's beampattern
        Case1 -- if the beamforming is a time domain beamforming, either ploar or catersian
                all the beam instance are plotted in the same figure, because it's frequency invariant.
        
        Case2-- if the beamforming is a frequency domain beamforming, then if polar, it will be plotted in 4 figures, 
                every figure represents a single frequency.
                if catesian, the figure number is equal to the beamforming instance number. One figure represents 
                a beamforming, the different frequency plotted in this same figure.
        
        Parameters
        ----------
        polar: bool True or False
        save_fig: bool True of False
        '''
        if self._beamforming_cout == 0:
            raise ValueError('No beamforming to analysis! Please append one firstly!')

        beamforming_dict = self._beamforming_list[0]
        coordinate = 'Polar' if polar else 'Cartesian'
        title = '{coordinate} Beampattern for {domain}Domain {dtype}Beamforming'.format(
                                coordinate=coordinate, 
                                domain = (beamforming_dict.get('beam_domain')).capitalize(), 
                                dtype = (beamforming_dict.get('beam_type')).capitalize())

        if polar:
            sweep_angle = np.concatenate((constants.get('angle_range'), 
                                        (constants.get('angle_range') + np.pi)), axis=0)
            xlim = (0, 2*np.pi)
        else:
            sweep_angle = np.degrees(constants.get('angle_range'))
            xlim = (0, 180)

        if beamforming_dict.get('beam_domain') == 'time':   
            plot = Plot(figsize=(6, 6)) if polar else Plot(figsize=(8, 5))
            plot.append_axes_combo(polar=polar)

            axis_limit_max = 0
            axis_limit_min = 0

            for beam_index, beam in enumerate(self._beamforming_list):
                response = beam['beam_instance'].beam_pattern()
                axis_limit_max = np.maximum(axis_limit_max, np.max(response))
                axis_limit_min = np.minimum(axis_limit_min, np.min(response))
                if polar:
                    response = np.concatenate((response, np.fliplr([response])[0]), axis=0)
                plot.plot_vector(response, sweep_angle, label= beam['beam_instance'].beam_label)
            plot.set_pipeline(xlim=xlim, ylim=(axis_limit_min-1, axis_limit_max+1), title=title,  
                                        xlabel='Azimuth Angle[Deg]', wait_for_user=False)
            plot.set_legend(loc=(0.7, 0.01), fontsize=6)

        elif beamforming_dict.get('beam_domain') == 'frequency':
            if polar:
                mini_freq_range  = constants.get('freq_range_mini')
                plot = Plot(figsize=(8, 8), subplot_cols=2, subplot_rows=2)
                plot.append_axes_combo(polar=True)

                # axis_limit_max = 0
                # axis_limit_min = 0
                for beam_index, beam in enumerate(self._beamforming_list):
                    for freq_index, freq in enumerate(mini_freq_range):
                        # get response 
                        response = beam['beam_instance'].beam_pattern(freq)
                        # axis_limit_max = np.maximum(axis_limit_max, np.max(response))
                        # axis_limit_min = np.minimum(axis_limit_min, np.min(response))
                        response = np.concatenate((response, np.fliplr([response])[0]), axis=0)
                        # plot
                        plot.plot_vector(response, sweep_angle, 
                                        label= beam['beam_instance'].beam_label, subplot_index=freq_index)
                        plot.set_pipeline(xlim=(0, 2*np.pi), ylim=(-50, 1), 
                                        title='', xlabel='Azimuth Angle[Deg]_{}Hz'.format(freq), 
                                        wait_for_user=False, subplot_index=freq_index)
                plot.set_legend(loc=(0.5, 0.01), fontsize=6, subplot_index=3)
                plot.suptitle(title)

            else:
                plot = Plot(figsize=(8, 4 * self._beamforming_cout), subplot_rows=self._beamforming_cout)
                plot.append_axes_combo()
                axis_limit_max = 0
                axis_limit_min = 0
                for beam_index, beam in enumerate(self._beamforming_list):
                    for freq in constants.get('freq_range_small'):
                        response = beam['beam_instance'].beam_pattern(freq)
                        axis_limit_max = np.maximum(axis_limit_max, np.max(response))
                        axis_limit_min = np.minimum(axis_limit_min, np.min(response))
                        plot.plot_vector(response, np.degrees(constants.get('angle_range')),
                                    label='{} kHz'.format( freq / 1000), subplot_index=beam_index)
                        plot.set_pipeline(title=beam['beam_instance'].beam_label, 
                                         xlim=xlim, ylim=(axis_limit_min-1, axis_limit_max+1), xlabel='Azimuth Angle[Deg]', 
                                         ylabel='Beampatten[dB]', wait_for_user=False,
                                         subplot_index=beam_index)
                plot.set_legend(subplot_index=self._beamforming_cout - 1)

                if self._beamforming_cout > 1:
                    plot.suptitle(title, 0.5)

        if save_fig:
            fig_name = self.fig_name('Beampattern_{coordinate}'.format(coordinate=coordinate))
            plot.save_figure(fig_name)
        
        plot.wait_for_user()


    def heatmap_beam_pattern(self, save_fig=True):
        '''
        beampattern heatmap only available for frequency domain beamforming
        case : draw beampatten for heatmap. 
               There will be as many subplots as beamforming numbers.
        '''
        if self._beamforming_cout == 0:
            raise ValueError('No beamforming to analysis! Please append one firstly!')
        
        beamforming_dict = self._beamforming_list[0]
        if beamforming_dict.get('beam_domain') == 'time':
            raise ValueError('Time domain beamforming donnot support heatmap plot!')

        print('This may take a few seconds. Please wating...')
        figsize = (10, 5) if self._beamforming_cout == 1 else (10, 3 * self._beamforming_cout)
        plot = Plot(figsize=figsize, subplot_rows=self._beamforming_cout)
        plot.append_axes_combo()

        xticks = np.arange(0, len(constants.get('angle_range')), 60)
        yticks = np.arange(0, 900, 100)
        xticklabels = np.arange(0, len(constants.get('angle_range')), 60, dtype=int) / 2.0
        yticklabels = np.linspace(8, 0, num=9, endpoint=True, dtype=int)

        freq_range = np.linspace(8000, 1, num=800, endpoint=True)
        for beam_index, beam in enumerate(self._beamforming_list):

            response_matrix = np.zeros((len(freq_range), len(constants.get('angle_range'))))
            for freq_index, freq in enumerate(freq_range):
                response_matrix[freq_index:] = beam['beam_instance'].beam_pattern(freq)

            plot.plot_heatmap(response_matrix, label_pad=-65, subplot_index=beam_index)
            plot.set_pipeline(title=beam['beam_instance'].beam_label, xticks=xticks, 
                            yticks=yticks, xticklabels=[int(tick) for tick in xticklabels], 
                            yticklabels=yticklabels, xlabel='Azimuth Angle[Deg]', 
                            ylabel='Freq[kHz]', wait_for_user=False, subplot_index=beam_index)

        if self._beamforming_cout > 1:
            plot.suptitle('Beampattern Heatmap Compare for Different Beamforming', hspace=0.5)

        if save_fig:
            fig_name = self.fig_name('Beampattern_Heatmap')
            plot.save_figure(fig_name)

        plot.wait_for_user()

    
    def __frequency_fixed_noise_improvement_performance(self, wng_or_df, save_fig=True):
        '''
        Only for frequency domain or 
        beamforming performace for directivity and white noise gain
        case : draw directivity or white noise gain in only one figure

        Parameter
        ---------
        wng_or_df: string
            White Noise Gain or Directivity
        '''
        freq_range = constants.get('freq_range_large')

        plot= Plot(figsize=(8, 6))
        plot.append_axes()
        axis_limit_max = []
        axis_limit_min = []
        for beam_index, beam in enumerate(self._beamforming_list):

            gain_array = np.zeros_like(freq_range, dtype=np.float32)
            for freq_index, freq in enumerate(freq_range):
                if wng_or_df == 'White Noise Gain':
                    gain_array[freq_index] = beam['beam_instance'].white_noise_gain(freq)
                elif wng_or_df == 'Directivity':
                    gain_array[freq_index] = beam['beam_instance'].directivity(freq)
                else:
                    raise ValueError('Please check DF or WNG are allowed')

            axis_limit_max.append(np.max(gain_array))
            axis_limit_min.append(np.min(gain_array))
            plot.plot_vector(gain_array, freq_range,  label=beam['beam_instance'].beam_label)

        if save_fig:
            fig_name = self.fig_name(wng_or_df)
        else:
            fig_name = None

        beamforming_dict = self._beamforming_list[0]
        title = '{wng_or_df} for {domain}Domain {dtype}Beamforming'.format(
                                wng_or_df=wng_or_df, 
                                domain = (beamforming_dict.get('beam_domain')).capitalize(), 
                                dtype = (beamforming_dict.get('beam_type')).capitalize())

        plot.set_pipeline(title=title, xlim=(0, 8000), ylim=(np.min(axis_limit_min) - 1, 
                        np.max(axis_limit_max) + 1), xlabel='Freq[Hz]', ylabel='dB', 
                        legend='lower right', fig_name=fig_name)
        

    def white_noise_gain(self, save_fig=True):
        
        if self._beamforming_cout == 0:
            raise ValueError('No beamforming to analysis! Please append one firstly!')
        
        if self._beamforming_list[0].get('beam_type') == 'adaptive':
            raise ValueError('Adaptive Beamforming donnot have white noise gain property!')

        if self._beamforming_list[0].get('beam_domain') == 'time':
            for beam in self._beamforming_list:
                wng = beam.get('beam_instance').white_noise_gain
                beam_str = beam.get('beam_instance').beam_label
                print(f'The White Noise Gain of {beam_str} is {wng}')

        if self._beamforming_list[0].get('beam_domain') == 'frequency':
            self.__frequency_fixed_noise_improvement_performance('White Noise Gain', save_fig=save_fig)
    

    def directivity(self, save_fig=True):

        if self._beamforming_cout == 0:
            raise ValueError('No beamforming to analysis! Please append one firstly!')

        if self._beamforming_list[0].get('beam_type') == 'adaptive':
            raise ValueError('Adaptive Beamforming donnot have white noise gain property!')

        if self._beamforming_list[0].get('beam_domain') == 'time':
            for beam in self._beamforming_list:
                di = beam.get('beam_instance').directivity
                beam_str = beam.get('beam_instance').beam_label
                print(f'The White Noise Gain of {beam_str} is {di}')

        if self._beamforming_list[0].get('beam_domain') == 'frequency':
            self.__frequency_fixed_noise_improvement_performance('Directivity', save_fig=save_fig)



    def noise_reduction_signal_distortion_factor(self):

        if self._beamforming_cout == 0:
            raise ValueError('No beamforming to analysis! Please append one firstly!')

        if self._beamforming_list[0].get('beam_type') == 'fixed':
            raise ValueError('Time Beamforming donnot have this property!')
        
        if self._beamforming_list[0].get('beam_domain') == 'time':
            attribute = 'performance'
        else:
            attribute = 'broadband_performance_dict'

        for index, beam in enumerate(self._beamforming_list, 1):
            print("************************************************")
            print('The performance of {index}th beamforming {label} is:'.format(index=index, 
                            label=beam.get('beam_instance').beam_label))
            performance = getattr(beam.get('beam_instance'), attribute)
            pprint(performance)