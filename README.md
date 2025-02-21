# PDA-AE
PDA
- MIMO通信
- 検出器にPDA
- 変調器は従来のQAMかニューラルネットワーク (NN) で構成か選べる
- 検出器は従来のPDAか適応スケールビリーフ (ASB) か深層展開 (DU)か選べる

![Image](https://github.com/user-attachments/assets/f3a9b51e-3ac4-4d4f-9371-2bf58fbbfd84)

## シミュレーションパラメータの説明
### 学習・テスト 共通設定
#### モード選択
- use_nnmod (bool) - FalseならQAM変調器，TrueならNN変調器になる
- ASB_mode (str) - 検出器の選択
  - "PDA" - 従来のPDA．
  - "ASB" - ASBを用いたPDA．繰り返し$i$回目におけるスケーリング係数 $\mu^{(i)}$は，最大繰り返し回数 $I$，調整パラメータ $d_1, d_2$を用いて次式で与えられる．<br>
  $$\mu^{(i)} = d_1 \left( \dfrac{i}{I} \right) ^ {d_2}, \quad i \in \\{1,2,\cdots,I\\}$$
  - "DU" &thinsp; - 深層展開を適用したPDA．初期値は上式で与えられる．
#### 基本設定
- M (int) - 送信アンテナ本数
- N (int) - 受信アンテナ本数
- Q_ant (int) - 1アンテナあたりの多値数．use_nnmod = False の場合， $2$または $4^n$ ($n$は任意の自然数) で指定．use_nnmod = True の場合は2以上の任意の自然数で指定．
- Kd (int) - データフレーム長
- num_joint_ant (int, float) - {0.5, 1} <br>
  - 0.5 - シンボルの実部（同相成分）と虚部（直行成分）を独立に検出
  - 1 &nbsp;&thinsp; - シンボルの実部と虚部を
- niter_PDA (int) - PDAの繰り返し回数 $I$
- d1 (float) - スケーリング係数の調整パラメータ $d_1$
- d2 (float) - スケーリング係数の調整パラメータ $d_2$

### 学習設定
#### 変調器・検出器 共通設定
- epochs (int) - エポック数
- mbs (int) - ミニバッチサイズ
- EsN0_train (float) - 学習時の $E_\mathrm{s} / N_0 \ [\mathrm{dB}]$
#### 変調器
- hidden_depth (int) - 中間層（隠れ層）の深さ
- hidden_dim (int) - 中間層（隠れ層）の次元
- ind_cons (bool) - Falseなら各アンテナの信号点配置は同じになり，Trueなら独立した信号点配置になる
<!--  -->
- lr_mod (float) - 変調器の学習率
- drop_start_mod (int) - 変調器の学習率スケジューラの減衰開始エポック
- drop_factor_mod (float) - 変調器の学習率スケジューラの減衰率
#### 検出器
- lr_det (float) - 検出器の学習率
- drop_start_det (int) - 検出器の学習率スケジューラの減衰開始エポック
- drop_factor_det (float) - 検出器の学習率スケジューラの減衰率

### テスト設定
- EsN0_test (list, ndarray) - テスト時の $E_\mathrm{s} / N_0 \ [\mathrm{dB}]$を並べた一次元配列
- nloop_max (int, float) - $E_\mathrm{s}/N_0$毎のシミュレーション回数．十分なエラー数が得られるまで計算をしたい場合はfloat('inf')に設定する．[^1]
- SE_max (int, float) - シンボルエラー数がこの設定値に達すると，その $E_\mathrm{s} / N_0$でのシミュレーションを打ち切る．早期終了しない場合はfloat('inf')に設定する．[^1]

### 並列処理設定
- nworker (int) - 並列ワーカー数（並列処理は計算機サーバーでの実行時のみ行う）

[^1]: nloop_maxとSE_maxを両方ともfloat('inf')に設定すると計算が終わらないので，片方は有限値に設定すること．
