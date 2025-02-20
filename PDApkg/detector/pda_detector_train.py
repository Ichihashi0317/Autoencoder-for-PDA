import torch
import torch.nn as nn


class PDA_detector_train(nn.Module):
    def __init__(self, SIM):
        super().__init__()
        # 変数取り込み
        self.SIM = SIM
        K = SIM.niter_PDA
        # シンボルエネルギー設定
        self.Es = 1.0
        # スケーリング係数
        if SIM.ASB_mode == 'none':
            self.mu = torch.ones(SIM.niter_PDA)
        elif SIM.ASB_mode in ('ASB', 'DU'):
            d1 = SIM.d1
            d2 = SIM.d2
            self.mu = (d1 / self.Es) * (torch.arange(1,K+1) / K)**d2
            if SIM.ASB_mode == 'DU':
                self.mu = nn.Parameter(self.mu)
    
    def forward(self, Y, H, x_rep):
        # 変数取り込み
        if self.SIM.num_joint_ant == 0.5:
            M = self.SIM.M * 2
            N = self.SIM.N * 2
            Q = self.SIM.Q_dim
            N0 = self.SIM.N0 / 2.0
            Es = self.Es / 2.0
            k_LL = 1.0
        elif self.SIM.num_joint_ant == 1:
            M = self.SIM.M
            N = self.SIM.N
            Q = self.SIM.Q_ant
            N0 = self.SIM.N0
            Es = self.Es
            k_LL = 2.0
            # 複素化
            Y = Y[:N] + 1j * Y[N:]
            H = H[:N, :M] + 1j * H[N:, :M]
            M_ = x_rep.shape[0] // 2
            x_rep = x_rep[:M_] + 1j * x_rep[M_:]
        K = self.SIM.Kd
        niter = self.SIM.niter_PDA
        # 送信信号候補
        x_rep = x_rep.T.reshape(Q,1,-1,1)    # (Q,1,M,1) = (M,Q).T.reshape(Q,1,M,1)
        xc_rep = x_rep.conj().clone()       # 共役複素数
        xx_rep = (x_rep * x_rep).abs()      # 絶対値2乗
        # スケーリング係数
        mu = self.mu

        # 事前計算
        H_H = H.H.clone()
        Sigma = N0 * torch.eye(N, dtype=H.dtype)
        # 高次元化
        Y = Y.T.unsqueeze(dim=-1)           # (K,N,1) = (N,K).T[:,:,newaxis]
        # 初期化
        SR = torch.zeros([K, M, 1], dtype=x_rep.dtype)      # ソフトレプリカ
        ER = torch.full([K, M, 1], Es)                      # エネルギーのレプリカ
        Delta_diag = torch.zeros([K, M, M], dtype=H.dtype)  # 対角行列
        
        # PDA 全時刻一括処理
        for idx_iter in range(niter):
            # soft canceller
            Y_tilde = Y - H @ SR                # (K,N,1) = (K,N,1) - (N,M) @ (K,M,1)
            # belief generator
            Delta = ER - (SR*SR).abs()          # (K,M,1)
            Delta_diag[:,range(M),range(M)] = Delta[:,:,0].to(H.dtype)  # (K,M,M) = diag((K,M,1))
            R = H @ Delta_diag @ H_H + Sigma    # (K,N,N) = (N,M) @ (K,M,M) @ (M,N) + (N,N)
            tmp = H_H @ torch.linalg.inv(R)     # (K,M,N) = (M,N) @ (K,N,N)
            # tmp = H_H @ (1/R)
            Gamma = (tmp.unsqueeze(dim=2) @ H.T.unsqueeze(dim=2))[:,:,:,0].real     # (K,M,1) = ((K,M,N)[:,:,newaxis,:] @ (N,M).T[:,:,newaxis])[:,:,:,0]
            tmp = tmp @ Y_tilde                 # (K,M,1) = (K,M,N) @ (K,N,1)
            LLV = mu[idx_iter] * k_LL / (1.0 - Gamma * Delta) * ((xc_rep * (tmp + Gamma * SR)).real - 0.5 * xx_rep * Gamma)    # 対数尤度 (Q,K,M,1)
            if idx_iter == niter - 1:
                break
            # replica generator
            LV = torch.exp(LLV - LLV.max(dim=0, keepdim=True).values)   # 尤度 (Q,K,M,1)
            PP = LV / LV.sum(dim=0, keepdim=True)   # 事後確率 (Q,K,M,1)
            SR = (x_rep  * PP).sum(dim=0)           # (K,M,1)
            ER = (xx_rep * PP).sum(dim=0)           # (K,M,1)
        
        return (LLV - LLV.max(dim=0, keepdim=True).values)[:,:,:,0].permute(1,2,0).reshape(K*M, Q)  # (K*M,Q) = (Q,K,M,1)[:,:,:,0].permute(1,2,0).reshape(K*M,Q)