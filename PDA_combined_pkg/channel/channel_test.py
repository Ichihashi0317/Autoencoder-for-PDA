import math
import numpy as np


class Channel_test():
    def __init__(self, SIM):
        self.SIM = SIM
        self.sigma_channel = 1.0 / math.sqrt(2.0)

    def __call__(self, X):
        # 変数取り込み
        M = self.SIM.M
        N = self.SIM.N
        sigma_noise = self.SIM.sigma_noise
        K = X.shape[1]
        # 通信路行列生成
        Hr = np.random.normal(scale=self.sigma_channel, size=[N, M])
        Hi = np.random.normal(scale=self.sigma_channel, size=[N, M])
        H = np.concatenate([np.concatenate([Hr, -Hi], axis=1), 
                            np.concatenate([Hi,  Hr], axis=1)], axis=0)
        # 雑音行列生成
        Z = np.random.normal(scale=sigma_noise, size=[2*N, K])
        # 伝搬
        Y = H @ X + Z
        return Y, H