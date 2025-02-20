import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .set_noise_param import set_noise_param
from .storage import Storage
from .modulator.modulator_joint import Modulator_joint
from .modulator.nnmod import NNMOD
from .channel.channel_train import Channel_train
from .detector.pda_detector_train import PDA_detector_train


class train_loop():
    def __init__(self, SIM, mod, nnmod, ch, det):
        # オブジェクト取り込み
        self.SIM = SIM
        self.mod = mod
        self.nnmod = nnmod
        self.ch = ch
        self.det = det
        # 変数計算
        self.M = int(SIM.M / SIM.num_joint_ant)
        self.Q = int(SIM.Q_ant ** SIM.num_joint_ant)
        self.K = SIM.Kd
    
    def gen_labels(self):
        # return torch.randint(self.Q, size=[self.M, self.K])
        return torch.randint(self.Q, size=[self.K * self.M])
    
    def run(self, label):
        label = label.reshape(self.K, self.M).T
        if self.nnmod is None:
            label_ = label.detach().numpy()
            Xd_ = self.mod.modulate(label_)
            Xd = torch.from_numpy(Xd_).float()
            x_rep = torch.from_numpy(self.mod.x_rep).float()
        else:
            Xd = self.nnmod(label)
            x_rep = self.nnmod.x_rep
        Yd, H = self.ch(Xd)
        LLV = self.det(Yd, H, x_rep)
        return LLV


def calc_dist(SIM, x):
    if SIM.num_joint_ant == 1:
        M = x.shape[0] // 2
        x = x[:M, :] + 1j * x[M:, :]
    dist = np.abs(x[:,np.newaxis,:] - x[:,:,np.newaxis])    # (M,Q,Q) = (M,1,Q) - (M,Q,1)
    min_dist = dist[dist != 0.0].min()
    return min_dist


def train_model(SIM, nnmod=None):
    time_start = time.perf_counter()

    ### 変数取り込み
    epochs = SIM.epochs
    mbs = SIM.mbs
    EsN0_train = SIM.EsN0_train

    ### 変数用意
    # 損失
    RES_loss = np.empty(epochs)
    loss_func = nn.CrossEntropyLoss()
    # 変調器
    mod = Modulator_joint(SIM)
    if SIM.use_nnmod and (nnmod is None):
        nnmod = NNMOD(SIM)
    if SIM.use_nnmod:
        RES_dist = np.empty(epochs)
        opt_mod = optim.Adam(nnmod.parameters(), lr=SIM.lr_mod)
        sch_mod = optim.lr_scheduler.MultiStepLR(opt_mod, 
                                                 milestones=list(range(SIM.drop_start_mod, SIM.epochs)),
                                                 gamma=SIM.drop_factor_mod)
    # 通信路
    ch = Channel_train(SIM)
    # 検出器
    det = PDA_detector_train(SIM)
    if SIM.ASB_mode == 'DU':
        RES_DU = np.empty([epochs, SIM.niter_PDA])
        opt_det = optim.Adam(det.parameters(), lr=SIM.lr_det)
        sch_det = optim.lr_scheduler.MultiStepLR(opt_det, 
                                                 milestones=list(range(SIM.drop_start_det, SIM.epochs)),
                                                 gamma=SIM.drop_factor_det)
    # その他
    model = train_loop(SIM, mod, nnmod, ch, det)
    EsN0vec = [EsN0_train] if type(EsN0_train) in (int, float) else EsN0_train
    interval = max(epochs // 10, 1)
    
    ### 訓練ループ
    for i in range(epochs):
        # 損失計算
        loss = torch.zeros(1)
        for EsN0 in EsN0vec:
            set_noise_param(SIM, EsN0)
            for _ in range(mbs):
                label = model.gen_labels()
                llv = model.run(label)
                loss += loss_func(llv, label)
        loss /= mbs * len(EsN0vec)
        # 更新
        RES_loss[i] = loss.item()
        loss.backward()
        if SIM.use_nnmod:
            opt_mod.step()
            opt_mod.zero_grad()
            sch_mod.step()
            RES_dist[i] = calc_dist(SIM, nnmod.x_rep.detach().numpy())
        if SIM.ASB_mode == 'DU':
            opt_det.step()
            opt_det.zero_grad()
            sch_det.step()
            RES_DU[i] = det.mu.clone().detach().numpy().astype(float)
        # 途中経過
        if i % interval == 0:
            print(i, end=' ')
    print()

    # 送信レプリカ・正規化層係数更新
    # 計算グラフから切り離す
    if SIM.use_nnmod:
        nnmod.set_replica()
        nnmod.detach_params()
    
    ### 出力変数格納
    RES = Storage()
    RES.loss = RES_loss
    if SIM.use_nnmod:
        RES.nnmod = nnmod
        RES.dist = RES_dist
    if SIM.ASB_mode == 'DU':
        RES.DU = RES_DU
    RES.time = time.perf_counter() - time_start
    return RES