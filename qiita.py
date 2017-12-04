#!/usr/bin/env python


# coding: utf-8

# # LSTMによる時系列解析をchainerで実装してみた
# 
# [**Brains Consulting, Inc.**](https://www.brains-consulting.co.jp/) でインターンをさせていただいている情報系のM1です。
# 2017/7から LSTM を用いた時系列予測を chainer で実装してきました。
# 今回扱った時系列データは、**需要データ** になります。
# 
# chainer **1.24.0** と結構古い version を使っているので注意です。
# 
# コードは、[hrsma2i/chainer-LSTM-time-series-forecasting](https://github.com/hrsma2i/chainer-LSTM-time-series-forecasting) に上がっています。
# コードレビューしていただけたら幸いです！

# ## 概要
# 
# - 「**deep learning を用いた時系列予測のモデルが、既存の時系列予測システム(ARIMA, カオス時系列解析 etc)に対し、同等以上の性能を持つか**」 を検証した。
# - また、 LSTMに対する、いくつかの前処理のうち、「**どの前処理の方法が良いか**」を実験の結果から決定した。

# ## はじめに
# 
# - 背景: 現在、需要予測はARIMAやカオス時系列解析が用いられているが、各商品の時系列ごとに model を学習させている。
# - deep learning を用いれば、1つの model で複数商品に対応できるのではないか。
# - 今回は、複数商品対応 model の構築の前段階として、1商品 model の性能が既存システムと同等以上かを検証する。

# ## 問題の定義
# 
# **task**: 時系列予測 **time forecasting**
# 
# **input**: 現在までの商品の需要データ
# 
# - 1データ(行): 1/4ヶ月の総需要数
# - 時系列の本数: 3商品、それぞれの時系列データ
# - 時系列長さ: 100~300
# 
# **output**: 3ヶ月先まで(`4*3 = 12` データ) の予測値
# 
# - 複数時刻の予測値を出すので、**multi-step** forecasting

# In[ ]:

import pandas as pd

# もとのデータ
pd.read_csv('data/airline_train.csv')


# ## Method
# 
# ### multi-step time forecasting
# 
# - **direct-recursive hybrid strategy**
#     - 次の予測値 $\hat{x}(t+2)$ を予測するのに、 前の予測値 $\hat{x}(t+1)$ も特徴量として含める手法。
#     - ARIMAなどのmodelを使う場合、入力数が変わるので、各時刻で異なる model を用意する必要があるが、LSTM は入力が可変なので、1つの model で済む。
#     
# ```python
# pred(t+1) = model(obs(0), ... , obs(t-1), obs(t))
# pred(t+2) = model(obs(0), ... , obs(t-1), obs(t), pred(t+1))
# pred(t+3) = model(obs(0), ... , obs(t-1), obs(t), pred(t+1), pred(t+2))
# ...
# ```
# 
# - multi-step には、以下のような他の手法もある。
# - c.f. [4 Strategies for Multi-Step Time Series Forecasting - Machine Learning Mastery](https://machinelearningmastery.com/multi-step-time-series-forecasting/)

# ### model architecture
# 
# ![LSTM](./LSTM.png)
# Fig.1: RNN（非展開時）
# 
# RNN 全体における、
# 
# - input shape: **(batch_size=1, timesteps, n_features=1)**
# - output shape: **(batch_size=1, timesteps, n_features=1)**
# 
# ![LSTM_deployed](./LSTM_deployed.png)
# Fig.2: RNN（展開時）
# 
# RNN の各時刻 t における、
# 
# - input shape: **(batch_size=1, n_features=1)**
# - output shape: **(batch_size=1, n_features=1)**
# 
# output は、input で入れた時系列に対し、1時刻ずれた同長の時系列が返る。
# 
# lossの計算、updateのタイミング: 各時刻の予測値と観測値のMSEの総和
# 
# $loss = \sum_{t=1}^{N_{train}} MSE(\hat{x}(t),x(t))$
# 
# **n_features** に関して、ARIMA などの Auto Regression な model は 前の時刻 t-1, t-2, ... などの値を **lag feature** 、特徴量の次元として扱うのに対し、LSTMでは、そういった lag の情報を **timestep** 軸として扱う。なので、今回、n_features=1。
# 
# 需要データ以外の時系列 (e.g. 気温など) も予測に使いたい場合は、その時系列データを特徴量の軸 n_features に加えればいい。
# 
# **batch_size** に関して、今回は、まだ 1 model で 1時系列 を学習させるので、 batch_size=1。ただ、 batch_size の増やし方がよくわからず、もし、複数商品に対応させる場合、言語モデルのように、**1つの商品の時系列を1つの文とみなして、batchを作るのがいいのだろうか**。このあたり、ご指摘いただけたら幸いです。
# 

# ## Preprocessing
# 
# 以下の前処理の必要性を検証すべく、7つの前処理の条件ごとに model を学習させた。
# 
# - **log scaling**
# - **differencing**
# - **MinMax scaling** $\in [-1, 1]$
#     - LSTM は tanh を含むので、[-1,1]の間にscalingが必要とうい記事を見たため。
#     - 変換に必要なparameterはtrainのみから算出
#     - c.f. [Time Series Forecasting with the Long Short-Term Memory Network in Python - Machine Learning Mastery](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)
# 
# ### dataset
# 
# 各時系列ごとに以下の3つに分割した。
# - test: 時系列の末尾 `12 = 3ヶ月*4` 時刻
# - train, val: test を除いた時系列 を train:val = 7:3 で分割

# ### 前処理の条件
# - **default**: log, diff, minmax
# - **not_log**: diff, minmax
# - **not_diff**: log, minmax
# - **minmax+**: log, diff, minmax $\in [0,1]$
# - **standard**: log, diff, StandardScaler （平均0, 分散1）
# - **not_label_scale**: log, diff, minmax（ただし、入力のみ）
# - **not_scale**: log, diff

# ### hyperparameter search
# 
# 各前処理の条件において、最も良いhyperparameterを選出する。
# 
# - search algorithm: random search
#     - NNの学習ではgrid searchより有効
#     - c.f. [James Bergstra - Random Search for Hyper-Parameter Optimization](http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
#     - n_epoch: 300 epoch/model
#     - n_sample: 100 model
# - metric: $min. mean_{e \in E_{best10}(hp)} (loss_{val}(e; hp))$: validation data に対する loss が最も低くなる 10 epoch の、各最小 loss の平均
#     - $loss_{val}(e; hp) = MSE_{t \in T_{val}}(\hat{y}_t(e; hp), x_{t+1})$: ある epoch における model の validation data に対する loss
#     - $\hat{y}_t(e; hp) = model(x_t; e; hp)$
#     - $E_{best10}(hp) = min10_{e \in E}(loss_{val}(e; hp))$: それが最小となるベスト10 epoch における $loss_{val, e}$ の値の組
#     - validation は shuffle なしの hold-out
# - search space:
#     - layer数: [1,5]
#     - 各layerのunit数: [1,10] layerによって変えた
#     - 最適化手法 (learning rate は default)
#         - SGD,
#         - Adam,
#         - RMSprop,
#         - AdaDelta,
#         - NesterovAG,
#         - MomentumSGD,

# ### select epoch
# 
# 各前処理の条件の、最も良かった hyper paramter において、最も良い epoch を選出する。
# 
# - metric: $min. loss_{val}(e; hp) = MSE_{t \in T_{val}}(\hat{y}_t(e; hp), x_{t+1})$: model の validation data に対する loss

# ### comparison metric
# 
# 各条件ごとの上記で選出されたモデルの比較に、以下の指標を用いた。
# 
# $|log(P/A)| \in [0, \infty)$
# - 小さいほどよい。
# 
# 理由として、現在のシステムの評価では、P/Aが用いられている。
# 
# P/A (predict/acutual) $\in [0, \infty)$
# - 1に近いほど良い。
# 
# また、P/Aが1より小さい model と、1以上の model を同時に評価できるように
# 
# $|log(P/A)| \in [0, \infty)$
# 
# を評価指標とした

# ### result
# 3商品すべてにおいて、not_logが最も良かった。よって、後述のARIMAとの比較では、**not_logの条件を用いたLSTM**と比較を行う。
# なお、1つの model を 300 epoch 学習させる場合、 2分ほどかかった。

# In[ ]:

# 詳細
from model_api import main_verification

prcs = [
    'not_log',
    'not_diff',
    'minmax+',
    'standard',
    'not_scale',
    'not_label_scale',
]
for name_prc in prcs:
    main_verification(name_prc=name_prc, verbose=False)


# ## Comparison with the current system
# 
# 以下の前処理の必要性を検証すべく、7つの前処理の条件ごとに model を学習させた。
# 
# ### dataset
# 各時系列ごとに以下の3つに分割した。
# - test: `12 = 3ヶ月*4=` (2016-04-01 ~ 2016-06-22) 
# - train, val: 元の時系列(~ 2016-3-22) を train:val = 7:3 で分割
# 
# ### comparison
# - **baseline**: 既存予測システム
# - **LSTM**
#     - preprocessing: not_log
#     - hyperparameter: 前述の選出方法
#     - epoch:  前述の選出方法
# 
# ### comparison metric
# - test data に対する、BciScore
# 
# ### result

# In[ ]:

from model_api import main_compare

name_prc='not_log'
main_compare(name_prc=name_prc)


# ## Conclusion
# 
# - 3商品中、1商品のみ、既存モデルに対し competitive な精度を出せた。
# - validation data に関しては結構いい精度だった。
#     - 予測階差を戻すときに、valでは1時刻前の観測値に予測階差を、testでは1時刻前の予測値に予測階差を足して戻しているので、その実装の部分がおかしい可能性がある。（code review していただけたら、嬉しいです。）
# - 前処理は、**階差、MinMax scaling[-1,1]** をすると精度があがることがわかった。ただし、**log scaling は良くならない**。

# ## Discussion, Future work 
# 
# - optimization method は 固定し、 learning rate を search したほうが良い結果が得られる可能性がある。
#     - RMSprop が今回、選ばれることが多かったので、RMSpropに固定
# - 複数商品への拡張
#     - 他の商品の時系列をsample軸に持ってきて、mini-batch learning をする。
#     - 表現力が必要なので、unit数やlayer数が多くなると考えられる。
#     - hyperparameterにbatch_sizeを追加する必要がある。
#     - LSTM用の regularization を適用する必要がある。
#         - Batch Norm for LSTM
#             - weight normalization
#             - reccurent BN
#             - layer normalization
#             - c.f. [ニューラルネット勉強会（LSTM編） chainer stateless/full BatchNorm Dropout Nstep](http://isw3.naist.jp/~neubig/student/2015/seitaro-s/161025neuralnet_study_LSTM.pdf)
#         - Dropout はLSTM blockすべてにつけるのではなく、LSTM内のある部分に施さないと意味がないらしい。
#             - zone out
#             - c.f. [ニューラルネット勉強会（LSTM編） chainer stateless/full BatchNorm Dropout Nstep](http://isw3.naist.jp/~neubig/student/2015/seitaro-s/161025neuralnet_study_LSTM.pdf)
# - truncated ver.
#     - 各商品の時系列の長さが200を超えてくるとbackpropの計算量が膨大になるのでEOSと`loss.unchain_backward`を用いて**TBPTT**を実行
#     - 
#     - stateful必須。EOSが出たらreset_state()
#     - LSTMでは、すべての時刻の入力を予測に使うことができるので、時系列を分けるときに使うwindowは、データがかぶらない**tumbling window**が最適と考えられる。
#         - c.f. [https://msdn.microsoft.com/en-us/library/dn835019.aspx](https://msdn.microsoft.com/library/en-us/Dn835055.aspx)
#     - 時系列の分ける長さは以下を参照（季節性を考慮し、1年という仮説が立てられる。）
#         - c.f. [How to Prepare Sequence Prediction for Truncated Backpropagation Through Time in Keras - Machine Learning Mastery](https://machinelearningmastery.com/truncated-backpropagation-through-time-in-keras/)
# - QRNN ver.
#     - LSTM の代わりに QRNN を用いて、学習時間を短縮できる可能性がある。
#     - timesteps数が多い処理なので有効と考えられる。
#     - c.f. 
#         - [LSTMを超える期待の新星、QRNN - Qiita](https://qiita.com/icoxfog417/items/d77912e10a7c60ae680e)
#         - [salesforce/pytorch-qrnn: PyTorch implementation of the Quasi-Recurrent Neural Network - up to 16 times faster than NVIDIA's cuDNN LSTM](https://github.com/salesforce/pytorch-qrnn) *"Speeds are between 2 and 17 times faster than NVIDIA's cuDNN LSTM"*

# ## 感想
# 時系列解析、この量のコーディングをしたは、初めてだったので、てんやわんやでした。
# コード、理論ともに拙い部分が多いです。特に、
# 
# - pre|post-processing
# - hyperparameter search
# 
# あたりが理論、実装ともに不安です。詳しい方、ご指摘ただけたら非常に幸いです。
# 
# [hrsma2i/chainer-LSTM-time-series-forecasting](https://github.com/hrsma2i/chainer-LSTM-time-series-forecasting) 

# ## Reference & Appendix
# 
# ### chainer tutorial
# - [Chainer: ビギナー向けチュートリアル Vol.1 - Qiita](https://qiita.com/mitmul/items/eccf4e0a84cb784ba84a) chainerは**training loopを抽象化するクラス**間の関係が煩雑なので、それらの**関係図**がわかりやすい。
#     - [Chainer v3 ビギナー向けチュートリアル - Qiita](https://qiita.com/mitmul/items/1e35fba085eb07a92560)
# - [Chainerのモデルのセーブとロード - 無限グミ](http://toua20001.hatenablog.com/entry/2016/11/15/203332)
# - [勤労感謝の日なのでChainerの勤労(Training)に感謝してextensionsを全部試した話 - EnsekiTT Blog](http://ensekitt.hatenablog.com/entry/2016/11/24/012539)

# ### LSTM 
# - [Machine Learning Mastery](https://machinelearningmastery.com/)
#     - **大変、お世話になりました。** 日本語だとNLPのものしかない中、LSTMを用いた時系列解析のkerasでのハンズオンが豊富にあります。特に以下のページから見るといいかもしれません。
#     - [Mini-Course on Long Short-Term Memory Recurrent Neural Networks with Keras - Machine Learning Mastery](https://machinelearningmastery.com/long-short-term-memory-recurrent-neural-networks-mini-course/)サイト内のLSTMに関する post が course 形式で順序だってまとめられている。
#     - [Time Series Forecasting with the Long Short-Term Memory Network in Python - Machine Learning Mastery](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/) kerasだが、preprocessing、testなどがまとまっている。
# - [LSTMネットワークの概要 - Qiita](https://qiita.com/KojiOhki/items/89cd7b69a8a6239d67ca)
# - [Fei-Fei Li & Justin Johnson & Serena Yeung Lecture 10: Recurrent Neural Networks](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture10.pdf)
#     - RNN の application による構造の分類、many-to-many, many-to-oneなどの4種類が絵で説明されててわかりやすい。
# - [ニューラルネット勉強会（LSTM編）](http://isw3.naist.jp/~neubig/student/2015/seitaro-s/161025neuralnet_study_LSTM.pdf)
# - [わかるLSTM ～ 最近の動向と共に - Qiita](https://qiita.com/t_Signull/items/21b82be280b46f467d1b)
# - [LSTMにsin波を覚えてもらう(chainer trainerの速習) - Qiita](https://qiita.com/chachay/items/052406176c55dd5b9a6a) 今回のコードのベースにさせていただきました。ただ、sin波にしか対応していなかったり、dataset,updater,loss,iteratorの定義が難しかったので、 [ここ](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/) を参考に改良。
# - [Keras - stateful vs stateless LSTMs](https://stackoverflow.com/questions/39681046/keras-stateful-vs-stateless-lstms)

# ### hyperparamter search
# あまり、体系だってhyperparameterについて書かれてる資料が見つからなかったです。コード付きの資料で実装面も勉強したいです。hyperoptなども試したのですが、うまく使いこなせませんでした。
# - [ディープラーニングにおける学習のコツとハイパーパラメータのガイドライン - Qiita](https://qiita.com/akimach/items/88356ef46e98f045b035)
# - [深層学習：ハイパーパラメータの設定に迷っている人へ - HELLO CYBERNETICS](http://s0sem0y.hatenablog.com/entry/2016/11/13/035443)
# - [LSTM: A Search Space Odyssey](https://arxiv.org/pdf/1503.04069.pdf)

# ### 設計
# この量のコーディングをしたのは初めてで、「うわ！なんでこんな書き方したんだ自分」って思うことが多く、全然綺麗にかけず、修正・追加したりするのが大変でした。この量じゃそんな設計するほどではないだろうと思われるかもしれませんが、ご容赦ください。精進いたします！pre|post-processing 周りが本当に難しく、何度も頭がこんがらがりました。
# 
# 全然うまく反映できていませんが、一応、下のslideを参考にさせていただきました。
# - [機械学習モデルの実装における、テストについて - Qiita](https://qiita.com/icoxfog417/items/67764a6756c4548b5fb8)
# - [機械学習で泣かないためのコード設計](https://www.slideshare.net/takahirokubo7792/ss-65413290)
