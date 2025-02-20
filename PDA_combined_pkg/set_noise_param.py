import math

def set_noise_param(SIM, EsN0_dB):
    EsN0 = 10.0**(EsN0_dB / 10.0)
    SIM.N0 = SIM.M / EsN0
    SIM.sigma_noise = math.sqrt(SIM.N0 / 2.0)