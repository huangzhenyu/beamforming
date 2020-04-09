from frequency_domain_beamforming.fixed_beamforming import Subspace

sp_beam = Subspace(2, 8, 0, 8)

sp_beam.beam_pattern_polar(1000)