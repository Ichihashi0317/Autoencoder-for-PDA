import math
import torch


class Channel_train():
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
        Hr = torch.normal(torch.zeros(N, M), std=self.sigma_channel)
        Hi = torch.normal(torch.zeros(N, M), std=self.sigma_channel)
        H = torch.cat([torch.cat([Hr, -Hi], dim=1), 
                       torch.cat([Hi,  Hr], dim=1)], dim=0)
        H *= math.sqrt(2*M*N) / H.norm('fro')   # 正規化
        # 雑音行列生成
        Z = torch.normal(torch.zeros(2*N, K), std=sigma_noise)
        # 伝搬
        Y = H @ X + Z
        return Y, H