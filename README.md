# ClipAndUniToStnet
画像特徴抽出モデルCLIPとUNIを用いて組織画像から遺伝子発現を予測する

# 1．CLIPとUNIを使えるようにする
## [CLIP](https://github.com/openai/CLIP)
上のリンクにあるGitHubのページの通りにCLIPをインストールするとimport clipでモデルが使用できるようになる。

## [UNI](https://github.com/mahmoodlab/UNI?tab=readme-ov-file)
上のリンクにあるGitHubのページの通りにUNIをインストールするとUNIモデルが使用できるようになる。（アカウントの登録が必要）

# 2．空間トランスクリプトームデータを取得する
データは先行研究にある[ST-Net](https://github.com/bryanhe/ST-Net)のものを使用した。ST-NetのGitHubページからデータをダウンロードするとdata/hist2tscript/Human_breast_cancer_in_situ_capturing_transcriptomicsというパス