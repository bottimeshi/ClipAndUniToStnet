import collections
import glob
import gzip
import os
import pandas
import pickle
import torch
import tqdm
import math
import numpy as np

class IdentityDict(dict):
    """This variant of a dict defaults to the identity function if a key has
    no corresponding value.

    https://stackoverflow.com/questions/6229073/how-to-make-a-python-dictionary-that-returns-key-for-keys-missing-from-the-dicti
    """
    def __missing__(self, key):
        return key
    
def load_results(output_dir):
    results = []
    file_list = [f for f in os.listdir(output_dir) if f.endswith('.pt') and f.startswith('result') and not f.endswith('single.pt')]
    with tqdm.tqdm(total=len(file_list), desc="Loading results") as pbar:
        for file_name in file_list:
            if file_name.startswith('result_') and not file_name.endswith('single.pt'):
                file_path = os.path.join(output_dir, file_name)
                patient, y_pred, y_test = torch.load(file_path, weights_only=True)
                results.append((patient, y_pred, y_test))
                pbar.update(1)
    
    return results

def calculate_correlations(results):
    correlations = {}
    for patient, y_pred, y_test in results:
        corr = []
        y_pred = y_pred.cpu()
        y_test = y_test.cpu()
        y_pred_np = y_pred.numpy().T
        y_test_np = y_test.numpy().T
        for i in range(y_pred_np.shape[0]):
            corr.append(np.corrcoef(y_pred_np[i], y_test_np[i])[0, 1])  # 各列の相関
        correlations[patient] = corr
    return correlations

def LoadNameData(inputX_dir, inputY_dir,names_dir, ens_dir, data_root):
    y = []

    #get patient and spot list
    print("Get patient and spot list")
    lists = glob.glob(inputX_dir + "/*_*.features.pt")
    patient = collections.defaultdict(list)
    with tqdm.tqdm(total=len(lists)) as pbar:
        for (p, s) in map(lambda x: x.split("/")[-1][:-12].split("_"), lists):
            patient[p].append(s)
            pbar.update()

    #load x data and spots(pt)
    print("Get image features data")
    spots = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            spots[p] = {}
            for s in patient[p]:
                spots[p][s] = []
                dataX = torch.load(inputX_dir + "/" + p + "_" + s + ".features.pt", weights_only=True)
                sorted_dataX = {k: v for k, v in sorted(dataX.items(), key=lambda item: item[0])}
                for key, value in sorted_dataX.items():
                    spots[p][s].append(key)
                pbar.update()
    
    #load y data(npz)
    print("Get counts data")
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            for s in patient[p]:
                with open(inputY_dir + "/" + p + "_" + s + ".counts.pkl", "rb") as f:
                    datay = pickle.load(f)
                sorted_datay = {k: v for k, v in sorted(datay.items(), key=lambda item: item[0])}
                filtered_datay = {}
                for i in spots[p][s]:
                    if i in sorted_datay:
                        filtered_datay[i] = sorted_datay[i]
                for _, value in filtered_datay.items():
                    y.append(torch.from_numpy(value))
                pbar.update()
    y = torch.stack(y)
    #select top 250 columns
    mean_values = torch.mean(y, dim=0)
    _, top_indices = torch.topk(mean_values, 250)
    y = y[:, top_indices]

    # load names data
    if not os.path.exists(names_dir + "/names.pkl"):
        # make save directry
        if not os.path.exists(names_dir):
            os.makedirs(names_dir)

        # Load count data 
        print("Load count data")
        data = {}
        with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
            for p in patient:
                data[p] = {}
                for s in patient[p]:
                    with gzip.open(data_root + "/" + "BC" + p[2:] + "_" + s + "_stdata.tsv.gz", "rb") as f:
                        data[p][s] = pandas.read_csv(f, sep="\t")
                    pbar.update()
    
         # Get counts per patients
        print("Get gene lists")
        gene_names = set()
        with tqdm.tqdm(total=sum(map(len, data.values()))) as pbar:
            for p in data:
                for s in data[p]:
                    gene_names = gene_names.union(set(data[p][s].columns.values[1:]))
                    pbar.update()
        gene_names = list(gene_names)
        gene_names = np.sort(gene_names)
        gene_names = [gene_names[i] for i in top_indices]
        
        # change gene names eng to symbol
        symbol = None
        try:
            with open(os.path.join(ens_dir, "ensembl.pkl"), "rb") as f:
                symbol = pickle.load(f)
        except FileNotFoundError:
            ensembl = pandas.read_csv(os.path.join(ens_dir, "ensembl.tsv"), sep="\t")
            symbol = IdentityDict()
            for (index, row) in ensembl.iterrows():
                symbol[row["Ensembl ID(supplied by Ensembl)"]] = row["Approved symbol"]
            with open(os.path.join(ens_dir, "ensembl.pkl"), "wb") as f:
                pickle.dump(symbol, f)
        gene_names = [symbol[g] for g in gene_names]

        with open(names_dir + "/names.pkl", "wb") as f:
            pickle.dump(gene_names, f)
    else:
        # Load names data 
        with open(names_dir + "/names.pkl", "rb") as f:
            gene_names = pickle.load(f)

    return gene_names

def main():
    inputX_dir = "/host/STCLIP/data/forTraining/imageFeatures"
    inputY_dir = "/host/STCLIP/data/forTraining/countsData"
    names_dir = "/host/STCLIP/data/forAnalysis"
    ens_dir = "/host/STCLIP/data/forTraining/ensembl"
    data_root = "/host/STCLIP/data/hist2tscript/Human_breast_cancer_in_situ_capturing_transcriptomics"
    output_dir = "/host/STCLIP/output/cross"  # 保存された結果のディレクトリ
    gene_name = "FASN"

    # 結果の読み込み
    results = load_results(output_dir)

    # 相関の計算
    correlations = calculate_correlations(results)
    for patient, corr in correlations.items():
        print(f"Patient: {patient}")
        print(f"correlation size: {len(corr)}")

    # 遺伝子の名前取得
    names = LoadNameData(inputX_dir, inputY_dir, names_dir, ens_dir, data_root)

    # 遺伝子名と相関の出力
    if not gene_name in names:
        print(f"Gene name {gene_name} not found.")
    else:
        gene_index = names.index(gene_name)
        sorted_correlations = dict(sorted(correlations.items(), 
                                  key=lambda item: (math.isnan(item[1][gene_index]), -item[1][gene_index] if not math.isnan(item[1][gene_index]) else float('inf'))))
        for patient, corr in sorted_correlations.items():
            print(f"Patient: {patient}")
            print(f"{gene_name}: {corr[gene_index]}")

if __name__ == "__main__":
    main()