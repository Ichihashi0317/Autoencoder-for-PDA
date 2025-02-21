![Image](https://github.com/user-attachments/assets/f3a9b51e-3ac4-4d4f-9371-2bf58fbbfd84)

# PDA-AE
PDA
- MIMO通信
- 検出器にPDA
- 変調器は従来のQAMかニューラルネットワーク (NN) で構成か選べる
- 検出器は従来のPDAか適応スケールビリーフ (ASB) か深層展開 (DU)か選べる

## シミュレーションパラメータの説明
### 学習・テスト 共通設定
- use_nnmod (bool) -
- ASB_mode (str) - 
  - "PDA"
  - "ASB"
  - "DU"
- M (int) - 
- N (int) - 
- Q_ant (int) - <br>
  1アンテナあたりの多値数． use_nnmod=Falseの場合は$2$ または $4^n$ ($n$は任意の自然数) で指定．
- Kd (int) - シンボル長
- num_joint_ant (int, float) - {0.5, 1}
- niter_PDA (int) - 
- d1 (float) - 
- d2 (float) -

### 学習設定
#### 変調器・検出器 共通設定
- epochs (int) - 
- mbs (int) - 
- EsN0_train (float) - 
#### 変調器
- hidden_depth (int) - 
- hidden_dim (int) - 
- ind_cons (bool) - 
<!--  -->
- lr_mod (float) - 
- drop_start_mod (int) - 
- drop_factor_mod (float) - 
#### 検出器
- lr_det (float) - 
- drop_start_det (int) - 
- drop_factor_det (float) - 

### テスト設定
- EsN0_test (ndarray) - 
- nloop_max (int, float) - 
- SIM.SE_max (int, float) - 

### 並列処理設定
- nworker (int) - 


  
  ### 変調メソッド
  入力：
  - bits (ndarray) - 送信ビットの2次元配列．1送信ベクトルに対応するビット列を列ベクトルとして含み，それを行方向へ並べた行列．1送信ベクトルあたりのビット数を $q = M \log_2{Q_\mathrm{ant}}$ とすると，サイズは $(q \times K)$．