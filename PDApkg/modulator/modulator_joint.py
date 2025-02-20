import math
import numpy as np
from .modulator import Modulator


class Modulator_joint(Modulator):
    def __init__(self, SIM):
        # 変数取り出し
        Q_ant = SIM.Q_ant
        Q_dim = SIM.Q_dim
        M = SIM.M
        num_joint_ant = SIM.num_joint_ant
        # 変数計算
        super().__init__(Q_ant)
        Q_ = int(Q_ant ** num_joint_ant)
        q_ = int(math.log2(Q_))
        self.Q_dim = Q_dim
        self.q_ = q_
        self.M_ = int(M / num_joint_ant)
        self.num_joint_ant = num_joint_ant
        # 送信レプリカ
        if num_joint_ant == 0.5:
            label = np.arange(Q_dim).reshape(1, Q_dim)
        elif num_joint_ant == 1:
            label = np.vstack([np.arange(Q_ant) % Q_dim, 
                               np.arange(Q_ant) // Q_dim])
        self.x_rep = self._modulate(label)
        # # ラベル (joint) -> シンボル (joint) 変換用 2次元配列
        # if num_joint_ant == 0.5:
        #     self.symtab_joint = self.x_rep[0,:]
        # elif num_joint_ant == 1:
        #     self.symtab_joint = self.x_rep[0,:] + 1j * self.x_rep[1,:]
        # ビット列 -> ラベル (joint) 変換用 重みベクトル
        self.weight_ = 2 ** np.arange(q_)
        # ラベル (joint) -> ビット列 変換用 2次元配列
        self.bittab_joint = np.empty([q_, Q_], dtype=int)
        tmp = np.arange(Q_)
        for i in range(q_):
            self.bittab_joint[i, :] = tmp % 2
            tmp //= 2
    
    def modulate(self, labels):
        if self.num_joint_ant == 1:
            labels = np.vstack([labels % self.Q_dim, 
                                labels // self.Q_dim])
        return self._modulate(labels)
    
    def bit2label(self, bits):
        return self.weight_ @ bits.reshape(self.M_, self.q_, bits.shape[1])  # (M_, K) = (q_,) @ (q_all , K).reshape(M_, q_, K)
    
    def label2bit(self, labels):
        # (q_all, K) = (q_, M_, K).transpose(0 <-> 1).reshape(q_all, K), (q_, M_, K) = (q_, Q_)[:, (M_, K)]
        return self.bittab_joint[:, labels].transpose(1, 0, 2).reshape(self.M_ * self.q_, labels.shape[1])
    
    def _modulate(self, labels):
        if self.Q_ant == 4:
            return (labels - self.mean) * self.k_mod
        else:
            return self.symtab_dim[labels]