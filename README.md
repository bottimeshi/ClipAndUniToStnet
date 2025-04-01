# ClipAndUniToStnet
画像特徴抽出モデルCLIPとUNIを用いて組織画像から遺伝子発現を予測する

# 1．CLIPとUNIを使えるようにする
## [CLIP](https://github.com/openai/CLIP)
上のリンクにあるGitHubのページの通りにCLIPをインストールするとimport clipでモデルが使用できるようになる。

## [UNI](https://github.com/mahmoodlab/UNI?tab=readme-ov-file)
上のリンクにあるGitHubのページの通りにUNIをインストールするとUNIモデルが使用できるようになる。（アカウントの登録が必要）

# 2．空間トランスクリプトームデータを取得する
データは先行研究にある[ST-Net](https://github.com/bryanhe/ST-Net)のものを使用した。ST-NetのGitHubページからデータをダウンロードし、data/hist2tscript/Human_breast_cancer_in_situ_capturing_transcriptomicsというパスに画像データ、座標データ、遺伝子発現データの三つの形式のファイルを置く。もし、ST-Netをクローンして使用する際はそのままでは動かないので以下の点で注意が必要。
- python3 setup.py install　で環境構築ができる。
- ST-Net/stnet/utils/ensembl.tsvが存在せず、プログラムが動かない。どこで入手したかは忘れたが、このプロジェクトに含めておく。
- Human_breast_cancer_in_situ_capturing_transcriptomicsを参照するファイル（ST-Net/stnet/cmd/prepare/spatial.pyなど）のパスやファイル名がほとんど間違っているので書き換えが必要
- csvファイル参照時の項目名が実際のファイルとソースコードで異なっているところがある
- データファイル名に含まれるBCやBTタグの種類がソースコード内ではBCのみを参照しているので、おそらくBTのタグのファイルもBCとして読み込む必要がある
- 座標データの座標とソースコード内で扱う座標が若干異なっている。座標データの座標を四捨五入してからint型に変換することで対応可能

# 3．CLIPまたはUNIを用いて画像パッチごとの画像特徴ベクトルを取得する
初めに、createtifs.shを実行することでtifファイルを作成する。
## CLIP
CLIPによる特徴ベクトルの保存はSTCLIP/source/imageToFeature.pyを実行することで行うことができる。

## UNI
UNIによる特徴ベクトルの保存はUNIST/source/imageToFeature.pyを実行することで行うことができる。

# 4．画像特徴量ベクトルから遺伝子発現量への変換を学習する
saveCountData.pyを実行することで遺伝子発現量をオブジェクトとして保存した後、学習を行う。
## CLIP
CLIPによる学習はSTCLIP/source/crossValidation.pyを実行することで行うことができる。また、その結果の表示はSTCLIP/source/analysis.pyを実行することで行うことができる

## UNI
UNIによる学習はUNIST/source/crossValidation.pyを実行することで行うことができる。また、その結果の表示はUNIST/source/analysis.pyを実行することで行うことができる
