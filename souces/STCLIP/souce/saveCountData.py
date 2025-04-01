import collections
import glob
import gzip
import numpy as np
import os
import pandas
import pickle
import tqdm

#data root directry
data_root = "/host/STCLIP/data/hist2tscript/Human_breast_cancer_in_situ_capturing_transcriptomics"

#save root directry
save_dir = "/host/STCLIP/data/forTraining/countsData"

def GetPSList():
    # Wildcard search for patients/sections
    images = glob.glob(data_root + "/*_*_*.jpg")

    # Dict mapping patient ID (str) to a list of all sections available for the patient (List[str])
    print("Get patient and spot list")
    patient = collections.defaultdict(list)
    with tqdm.tqdm(total=len(images)) as pbar:
        for (p, s) in map(lambda x: x.split("/")[-1][3:-4].split("_"), images):
            patient[p].append(s)
            pbar.update()
    return patient

def LoadCountData(patient):
    # make save directry
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load count data 
    print("Load count data")
    data = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            data[p] = {}
            for s in patient[p]:
                file_root = data_root + "/" + p + "_" + s
                if newer_than(data_root + "/" + "BC" + p[2:] + "_" + s + "_stdata.tsv.gz", file_root + ".stdata.pkl") or not os.path.exists(file_root + ".stdata.pkl"):
                    with gzip.open(data_root + "/" + "BC" + p[2:] + "_" + s + "_stdata.tsv.gz", "rb") as f:
                        data[p][s] = pandas.read_csv(f, sep="\t")
                    with open(file_root + ".stdata.pkl", "wb") as f:
                        pickle.dump(data[p][s], f)
                else:
                    with open(file_root + ".stdata.pkl", "rb") as f:
                        data[p][s] = pickle.load(f)
                pbar.update()

    # Get counts per patients
    print("Get gene lists")
    section_header = None
    gene_names = set()
    with tqdm.tqdm(total=sum(map(len, data.values()))) as pbar:
        for p in data:
            for s in data[p]:
                section_header = data[p][s].columns.values[0]
                gene_names = gene_names.union(set(data[p][s].columns.values[1:]))
                pbar.update()
    gene_names = list(gene_names)
    gene_names.sort()
    gene_names = [section_header] + gene_names

    print("Get counts per spot")
    counts = {}
    with tqdm.tqdm(total=sum(map(len, data.values()))) as pbar:
        for p in data:
            counts[p] = {}
            for s in data[p]:
                counts[p][s] = {}
                # In the original data, genes with no expression in a section are dropped from the table.
                # This adds the columns back in so that comparisons across the sections can be done.
                missing = list(set(gene_names) - set(data[p][s].keys()))
                c = data[p][s].values[:, 1:].astype(float)
                pad = np.zeros((c.shape[0], len(missing)))
                c = np.concatenate((c, pad), axis=1)
                names = np.concatenate((data[p][s].keys().values[1:], np.array(missing)))
                c = c[:, np.argsort(names)]
                for (j, row) in data[p][s].iterrows():
                    counts[p][s][row.values[0]] = c[j, :]
                
                #save counts data
                with open(save_dir + "/" + p + "_" + s + ".counts" + ".pkl", "wb") as f:
                    pickle.dump(counts[p][s], f)
                pbar.update()

def newer_than(file1, file2):
    """
    Returns True if file1 is newer than file2.
    A typical use case is if file2 is generated using file1.
    For example:

    if newer_than(file1, file2):
        # update file2 based on file1
    """
    return os.path.isfile(file1) and (not os.path.isfile(file2) or os.path.getctime(file1) > os.path.getctime(file2))

def main():
    patient = GetPSList()
    LoadCountData(patient)

main()