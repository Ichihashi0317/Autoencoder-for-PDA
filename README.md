# Autoencoder-for-PDA

変調器をニューラルネットワークで構成し，PDA検出器に深層展開を適用した通信系のシミュレーションプログラム

## 概要
- 通信路はレイリーフェージング環境のMIMO通信路
- 変調器は，QAMか，ニューラルネットワーク (NN) で構成した変調器のどちらかを選択可能
- 検出器は，従来のPDAか，適応スケールビリーフ (ASB) あるいは深層展開 (DU) を適用したPDAのいずれかを選択可能
- 学習経過およびビット誤り率 (BER) 特性とシンボル誤り率 (SER) 特性を表示

![Image](https://github.com/user-attachments/assets/f3a9b51e-3ac4-4d4f-9371-2bf58fbbfd84)

### シミュレーションの流れ
- 送信機 <br>
送信ビット列がランダム生成され，変調器に入力される．
変調器は入力ビット列に対応するシンボル（変調信号を複素表示したもの）を出力する．
- 通信路 <br>
送信信号は通信路を介して受信機へ届く．
本シミュレーションは，複数の送信アンテナと受信アンテナを用いて同一時刻に同一周波数を使用して通信するMIMO通信を想定している．
そのため，送信信号はフェージング（信号強度の変化）に加え，他の送信アンテナからの送信信号と相互干渉した状態で受信機へ届く．
- 受信機 <br>
受信機内で雑音が加算された受信信号が検出器に入力される．
検出器では信号の分離・検出が行われ，送信ビットの判定値が出力される．
- 誤りビット数のカウント <br>
送信機への入力ビット列と受信機からの出力ビット列を比較し，誤りビット数をカウントする．
この処理を送信ビット列を変えながら繰り返し行い，平均ビット誤り率を算出する．

### 変調器と検出器の選択候補
変調器は以下の2つのどちらかを選択可能．
- QAM
- NN変調器 - NNで構成した変調器．受信機での検出誤りを最小化するようにシンボル配置が最適化される．

検出器は以下の3つのいずれかを選択可能．
- 従来のPDA
- PDA-ASB - ASB（PDAの検出過程で算出される送信信号の確率に対して補正係数を乗積）を用いたPDA．
- PDA-DU - ASBの補正係数を深層学習の一種であるDUで最適化したPDA検出．

NN変調器とPDA-DU検出器は，送信機から受信機までの一連の流れを自己符号化器として構成し，受信機出力が送信機入力に近づくように最適化される．

### プログラムの流れ
1. シミュレーション条件と学習条件の設定
2. NN変調器・PDA-DU検出器の学習（NN変調器またはPDA-DU検出器を選択した場合） <br> 学習後は学習経過を表示
4. 選択した変調器と検出器の性能テスト <br> テスト後はBER特性とSER特性を表示

詳細は文献[^MyPaper]を参照．


## シミュレーションパラメータの説明
### 学習・テスト 共通設定
#### モード選択
- use_nnmod (bool) - FalseならQAM変調器，TrueならNN変調器になる
- ASB_mode (str) - 検出器の選択
  - "PDA" - 従来のPDA．
  - "ASB" - PDA-ASB．
  - "DU" - PDA-DU．初期値はASBのスケーリング係数の設定値と同じになる．
  <!-- - "ASB" - ASBを用いたPDA．繰り返し $i$回目におけるスケーリング係数 $\mu^{(i)}$は，最大繰り返し回数 $I$，調整パラメータ $d_1, d_2$を用いて次式で与えられる．
  <br>
  $$\mu^{(i)} = d_1 \left( \dfrac{i}{I} \right) ^ {d_2}, \quad i \in \\{1,2,\cdots,I\\}$$
  - "DU" &thinsp; - DUを適用したPDA．初期値は上式で与えられる． -->

#### 基本設定
- M (int) - 送信アンテナ本数
- N (int) - 受信アンテナ本数
- Q_ant (int) - <br> 
1アンテナあたりの多値数．use_nnmodがFalseの場合， $2$または $4^n$ ($n$は任意の自然数) で指定．Trueの場合は2以上の任意の自然数で指定．
- Kd (int) - データフレーム長
- num_joint_ant (int, float) - NN変調時および信号検出時のアンテナ結合本数．現在0.5または1のみ指定可能．
  - 0.5 - シンボルの実部（同相成分）と虚部（直交成分）を別々に検出する．NN変調器の場合，格子状の信号点配置になる．
  - 1 &nbsp;&thinsp; - シンボルの実部と虚部を合わせて検出する．NN変調器の場合，非格子状の信号点配置になる．
- niter_PDA (int) - PDAの繰り返し回数
- d1 (float) - ASBのスケーリング係数の調整パラメータ[^MyPaper][^TakahashiIEEE]．ASB_modeが"ASB"または"DU"のときのみ指定可能．
- d2 (float) - ASBのスケーリング係数の調整パラメータ[^MyPaper][^TakahashiIEEE]．ASB_modeが"ASB"または"DU"のときのみ指定可能．

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
- nloop_max (int, float) - $E_\mathrm{s}/N_0$毎のシミュレーション回数 <br>
シンボルエラー数がSE_max（後述）に達するまで計算を継続する場合はfloat('inf')に設定する[^NotInf]．
- SE_max (int, float) - 早期終了条件 <br>
シンボルエラー数がこの設定値に達すると，その $E_\mathrm{s} / N_0$でのシミュレーションを打ち切る．早期終了しない場合はfloat('inf')に設定する[^NotInf]．

### 並列処理設定
- nworker (int) - 並列ワーカー数（並列処理は計算機サーバーでの実行時のみ行う）

[^MyPaper]: https://www.ieice.org/publications/ken/summary.php?contribution_id=128665&society_cd=CS&ken_id=CS&year=2024&presen_date=2024-01-18&schedule_id=8121&lang=jp&expandable=1
[^TakahashiIEEE]: https://ieeexplore.ieee.org/abstract/document/8543847
[^NotInf]: nloop_maxとSE_maxの両方をfloat('inf')に設定すると計算が終了しなため，どちらか一方は有限値に設定すること．
