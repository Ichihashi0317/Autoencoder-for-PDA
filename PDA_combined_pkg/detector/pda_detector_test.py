import numpy as np


class PDA_detector_test():
    def __init__(self, SIM, x_rep, mu=None):
        # 変数取り込み
        self.SIM = SIM
        # シンボルエネルギー設定
        self.Es = 1.0
        # スケーリング係数
        if SIM.ASB_mode == 'none':
            self.mu = np.ones(SIM.niter_PDA)
        elif SIM.ASB_mode == 'ASB':
            K = SIM.niter_PDA
            d1 = SIM.d1
            d2 = SIM.d2
            self.mu = (d1 / self.Es) * (np.arange(1,K+1) / K)**d2
        elif SIM.ASB_mode == 'DU':
            self.mu = mu
        # 送信信号候補
        if SIM.num_joint_ant == 0.5:
            M = x_rep.shape[0]
            Q = SIM.Q_dim
            xc_rep = x_rep
        elif SIM.num_joint_ant == 1:
            M = x_rep.shape[0] // 2
            Q = SIM.Q_ant
            x_rep = x_rep[:M, :] + 1j * x_rep[M:, :]
            xc_rep = x_rep.conj().copy()
        self.x_rep = x_rep.T.reshape(Q,1,M,1)           # (Q,1,M,1) = (M,Q).T.reshape(Q,1,M,1), QAMの場合はM = 1
        self.xc_rep = xc_rep.T.reshape(Q,1,M,1)         # 共益複素数
        self.xx_rep = np.abs(self.x_rep * self.x_rep)   # 絶対値2乗
    
    def __call__(self, Y, H):
        # 変数取り込み
        if self.SIM.num_joint_ant == 0.5:
            M = self.SIM.M * 2
            N = self.SIM.N * 2
            N0 = self.SIM.N0 / 2.0
            Es = self.Es / 2.0
            k_LL = 1.0
        elif self.SIM.num_joint_ant == 1:
            M = self.SIM.M
            N = self.SIM.N
            N0 = self.SIM.N0
            Es = self.Es
            k_LL = 2.0
            # 複素化
            Y = Y[:N] + 1j * Y[N:]
            H = H[:N, :M] + 1j * H[N:, :M]
        K = self.SIM.Kd
        niter = self.SIM.niter_PDA
        # 送信信号候補
        x_rep = self.x_rep
        xc_rep = self.xc_rep
        xx_rep = self.xx_rep
        # スケーリング係数
        mu = self.mu

        # 事前計算
        H_H = H.T.conj().copy()
        Sigma = N0 * np.eye(N, dtype=H.dtype)
        # 高次元化
        Y = Y.T[:,:,np.newaxis]     # (K,N,1) = (N,K).T[:,:,newaxis]
        # 初期化
        SR = np.zeros([K, M, 1], dtype=x_rep.dtype)     # ソフトレプリカ
        ER = np.full([K, M, 1], Es)                     # エネルギーのレプリカ
        Delta_diag = np.zeros([K, M, M], dtype=H.dtype) # 対角行列

        # PDA 全時刻一括処理
        for idx_iter in range(niter):
            # soft canceller
            Y_tilde = Y - H @ SR        # (K,N,1) = (K,N,1) - (N,M) @ (K,M,1)
            # belief generator
            Delta = ER - np.abs(SR*SR)  # (K,M,1)
            Delta_diag[:,range(M),range(M)] = Delta[:,:,0]  # (K,M,M) = diag((K,M,1))
            R = H @ Delta_diag @ H_H + Sigma    # (K,N,N) = (N,M) @ (K,M,M) @ (M,N) + (N,N)
            tmp = H_H @ np.linalg.inv(R)        # (K,M,N) = (M,N) @ (K,N,N)
            Gamma = (tmp[:,:,np.newaxis,:] @ H.T[:,:,np.newaxis])[:,:,:,0].real # (K,M,1) = ((K,M,N)[:,:,newaxis,:] @ (N,M).T[:,:,newaxis])[:,:,:,0]
            tmp = tmp @ Y_tilde                 # (K,M,1) = (K,M,N) @ (K,N,1)
            LLV = mu[idx_iter] * k_LL / (1.0 - Gamma * Delta) * ((xc_rep * (tmp + Gamma * SR)).real - 0.5 * xx_rep * Gamma) # 対数尤度 (Q,K,M,1)
            if idx_iter == niter - 1:
                break
            # replica generator
            LV = np.exp(LLV - LLV.max(axis=0, keepdims=True))   # 尤度　　 (Q,K,M,1)
            PP = LV / LV.sum(axis=0, keepdims=True)             # 事後確率 (Q,K,M,1)
            SR = (x_rep  * PP).sum(axis=0)  # (K,M,1)
            ER = (xx_rep * PP).sum(axis=0)  # (K,M,1)

        # 送信信号検出値
        # X_hat = SR[:,:,0].T                 # (M,K) = (K,M,1)[:,:,0].T
        # 送信信号検出値のラベル
        label = LLV.argmax(axis=0)[:,:,0].T # (M,K) = (Q,K,M,1).argmax(axis=0)[:,:,0].T
        
        return label