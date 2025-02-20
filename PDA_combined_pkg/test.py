from multiprocessing import Pool
import time
import math
import numpy as np

from .set_noise_param import set_noise_param
from .storage import Storage
from .modulator.modulator_joint import Modulator_joint
from .channel.channel_test import Channel_test
from .detector.pda_detector_test import PDA_detector_test


def test_loops(SIM, mod, nnmod, ch, det):
    
    # 変数取り込み
    M = SIM.M
    Kd = SIM.Kd
    nloop_max = SIM.nloop_max
    SE_max = SIM.SE_max
    q = int(math.log2(SIM.Q_ant)) * M
    if SIM.env_server:
        nloop_max /= SIM.nworker
    
    ### テストループ
    idx_loop = 0
    noe = np.zeros(2, dtype=int)

    while (idx_loop < nloop_max) and (noe[1] < SE_max):
        
        # 送信ビット
        bits = mod.gen_bits(M, Kd)

        # ラベルへ変換
        labels = mod.bit2label(bits)

        # 変調
        if nnmod is None:
            Xd = mod.modulate(labels)
        else:
            Xd = nnmod.test(labels)

        # 通信路
        Yd, H = ch(Xd)

        # 検出
        labels_hat = det(Yd, H)
        
        # 復調・硬判定
        bits_hat = mod.label2bit(labels_hat)

        # エラー数カウント
        iserror = (labels != labels_hat).any(axis=0)
        noe[0] += (bits[:, iserror] != bits_hat[:, iserror]).sum()
        noe[1] += iserror.sum()
        
        idx_loop += 1
    
    # 全送信データ数
    nod = np.array([q * Kd * idx_loop, Kd * idx_loop], dtype=np.int64)
    
    return idx_loop, noe, nod


def test_model(SIM, nnmod=None, DU_param=None):
    time_start = time.perf_counter()

    # 変数取り込み
    EsN0vec = SIM.EsN0_test

    ### テスト用インスタンス生成
    # 変復調器
    mod = Modulator_joint(SIM)
    if nnmod is None:
        x_rep = mod.x_rep
    else:
        nnmod.detach_params()
        x_rep = nnmod.x_rep.detach().numpy().astype(float)
    # 通信路
    ch = Channel_test(SIM)
    # PDA検出器
    det = PDA_detector_test(SIM, x_rep, DU_param)
    
    ### テスト
    ER = np.zeros([len(EsN0vec), 2])
    time_cusum = 0
    
    for idx_EsN0 in range(len(EsN0vec)):
        time_loops = -time.perf_counter()

        # Es/N0設定
        EsN0 = EsN0vec[idx_EsN0]
        set_noise_param(SIM, EsN0)
        
        # エラー数を計算
        if (not SIM.env_server) or (SIM.nworker == 1):
            nloop, noe, nod = test_loops(SIM, mod, nnmod, ch, det)
        else:
            with Pool(SIM.nworker) as p:
                res_list = p.starmap(test_loops, [(SIM, mod, nnmod, ch, det) for _ in range(SIM.nworker)])
            nloop = sum([res[0] for res in res_list])
            noe   = sum([res[1] for res in res_list])
            nod   = sum([res[2] for res in res_list])

        # EsN0ごとにエラーレートを格納
        ER[idx_EsN0, :] = noe / nod

        # 時間計測
        time_loops += time.perf_counter()
        time_cusum += time_loops

        # テスト経過表示
        print(EsN0,"dB, nloop =",nloop,", noe =",noe,", ER =",ER[idx_EsN0, :].astype(np.float16),",",np.array(time_loops, dtype=np.float16),"sec,",np.array(time_cusum, dtype=np.float16),"sec")

        # SER=0ならば早期終了
        if noe[1] == 0:
            print("break")
            break

    ### 出力変数格納
    RES = Storage()
    RES.BER = ER[:, 0]
    RES.SER = ER[:, 1]
    RES.time = time.perf_counter() - time_start
    return RES