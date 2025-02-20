import math
# import numpy as np
import torch
import torch.nn as nn


class Normalization_Layer():    # 実数等価
    def __init__(self):
        self.normalize = self.average_vector

    def average_vector(self, x, upd_nml_fact):
        if upd_nml_fact:
            self.nml_fact = 1.0 / (2.0 * (x**2).mean()).sqrt()
        return x * self.nml_fact


class NNMOD(nn.Module):
    def __init__(self, SIM):
        super().__init__()

        # 変数取り込み
        self.SIM = SIM
        self.hidden_depth = SIM.hidden_depth

        # 変数計算
        M_ = int(SIM.M / SIM.num_joint_ant)
        Q_ = int(SIM.Q_ant ** SIM.num_joint_ant)
        self.label_rep = torch.arange(Q_).expand(M_, -1)
        self.Eye = torch.eye(Q_)

        # NN構築
        layers_dim = [Q_] + [SIM.hidden_dim] * SIM.hidden_depth + [int(2 * SIM.num_joint_ant)]
        ncons = M_ if SIM.ind_cons else 1
        self.Wlist = nn.ParameterList()
        self.blist = nn.ParameterList()
        for i in range(SIM.hidden_depth + 1):
            ninput = layers_dim[i]
            noutput = layers_dim[i+1]
            lim = math.sqrt(6.0 / ninput)
            self.Wlist += [self._rnd_uniform(low=-lim, high=lim, size=[ncons, ninput, noutput])]
            self.blist += [self._rnd_uniform(low=-lim, high=lim, size=[ncons, 1, noutput])]
        
        # 正規化層
        self.nml = Normalization_Layer()
    
    def forward(self, label):
        # 正規化係数計算・送信レプリカ生成
        self.set_replica()
        # 送信シンボル生成
        x = self._forward(label)    # (2M, K)
        return x
    
    def test(self, label):
        # 送信シンボル生成
        with torch.no_grad():
            label = torch.from_numpy(label).long()  # (M_, K)
            x = self._forward(label)                # (2M, K)
        return x.detach().numpy().astype(float)
    
    def set_replica(self):
        self.x_rep = self._forward(self.label_rep, upd_nml_fact=True)   # (2M, Q_)

    def detach_params(self):
        self.x_rep = self.x_rep.detach()
        self.nml.nml_fact = self.nml.nml_fact.detach()
    
    @staticmethod
    def _rnd_uniform(low=0, high=1, size=(1,)):
        return torch.empty(size).uniform_(low, high)
    
    def _forward(self, label, upd_nml_fact=False):
        # 変数取り込み
        M = self.SIM.M

        # one-hotベクトルに変換
        x = self.Eye[label, :]  # (M_, K, Q_) = (Q_, Q_)[(M_, K), :]

        # NN順伝播
        for i in range(self.hidden_depth):
            x = torch.relu(x @ self.Wlist[i] + self.blist[i])   # 中間層
        x = x @ self.Wlist[-1] + self.blist[-1] # 出力層 (M_, K, 2*num_joint_ant)

        # 整形
        x = x.permute(2, 0, 1).reshape(2*M, -1)  # (2M, K)

        # 正規化
        x = self.nml.normalize(x, upd_nml_fact)
        
        return x