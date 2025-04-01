import collections
import glob
import gzip
import openslide
import os
import pandas
import pickle
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm
import tqdm

# 画像ディレクトリ
image_dir = "/host/STCLIP/data/hist2tscript/Human_breast_cancer_in_situ_capturing_transcriptomics"
# 特徴量を保存するディレクトリ
save_dir = "/host/UNIST/data/forTraining/imageFeatures"  # 保存先ディレクトリを指定

def imageToFeature(patient, image_patch):
    # 保存先ディレクトリが存在しない場合は作成
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # モデルをロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_dir = "/host/UNIST/assets/ckpts/uni2-h"
    timm_kwargs = {
       'model_name': 'vit_giant_patch14_224',
       'img_size': 224, 
       'patch_size': 14, 
       'depth': 24,
       'num_heads': 24,
       'init_values': 1e-5, 
       'embed_dim': 1536,
       'mlp_ratio': 2.66667*2,
       'num_classes': 0, 
       'no_embed_class': True,
       'mlp_layer': timm.layers.SwiGLUPacked, 
       'act_layer': torch.nn.SiLU, 
       'reg_tokens': 8, 
       'dynamic_img_size': True
      }
    model = timm.create_model(**timm_kwargs)
    model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
    model = model.to(device)
    model = nn.DataParallel(model)
    transform = transforms.Compose(
     [
      transforms.Resize(224),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ]
    )

    print("save image features")
    image_features = {}
    image_featuresList = []
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            image_features[p] = {}
            for s in patient[p]:
                image_features[p][s] = {}
                for key, value in image_patch[p][s].items():
                    # 画像を読み込み、前処理を行う
                    image = transform(value).unsqueeze(dim=0).to(device)

                    # 画像をCLIPに入力し、特徴量を計算
                    with torch.inference_mode():
                        image_features[p][s][key] = model(image)
                    image_featuresList.append(image_features[p][s][key])
                    
                    #メモリの開放
                    del image
                    torch.cuda.empty_cache()

                # 特徴量をptファイルとして保存
                pt_filename = p + "_" + s + ".features.pt"
                pt_path = os.path.join(save_dir, pt_filename)
                torch.save(image_features[p][s], pt_path)
                pbar.update()
    return image_featuresList
    
def GetPSList():
    # Wildcard search for patients/sections
    images = glob.glob(image_dir + "/*_*_*.tif")

    # Dict mapping patient ID (str) to a list of all sections available for the patient (List[str])
    print("Get patient and spot list")
    patient = collections.defaultdict(list)
    with tqdm.tqdm(total=len(images)) as pbar:
        for (p, s) in map(lambda x: x.split("/")[-1][3:-4].split("_"), images):
            patient[p].append(s)
            pbar.update()
    return patient

def LoadImageData(patient, window=224):
    print("Load tif files with openslide")
    slides = collections.defaultdict(dict)
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            for s in patient[p]:
                slides[p][s] = openslide.open_slide(image_dir + "/HE_{}_{}.tif".format(p, s))
                pbar.update()
    
    print("Load pixel Data")
    pixels = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            pixels[p] = {}
            for s in patient[p]:
                file_root = image_dir + "/" + p + "_" + s
                if newer_than(image_dir + "/" + "spots" + "_" + p + "_" + s + ".csv.gz", file_root + ".spots.pkl") or not os.path.exists(file_root + ".spots.pkl"):
                    with gzip.open(image_dir + "/" + "spots" + "_" + p + "_" + s + ".csv.gz", "rb") as f:
                        pixels[p][s] = pandas.read_csv(f, sep=",")
                    with open(file_root + ".spots.pkl", "wb") as f:
                        pickle.dump(pixels[p][s], f)
                else:
                    with open(file_root + ".spots.pkl", "rb") as f:
                        pixels[p][s] = pickle.load(f)
                pbar.update()

    print("Get patch per spot")
    orig_window = window
    image_patch = {}
    with tqdm.tqdm(total=sum(map(len, patient.values()))) as pbar:
        for p in patient:
            image_patch[p] = {}
            for s in patient[p]:
                image_patch[p][s] = {}
                slide = slides[p][s]
                for (_, row) in pixels[p][s].iterrows():
                    pixel = [int(round(row["X"])), int(round(row["Y"]))]
                    X = slide.read_region((pixel[0] - orig_window // 2, pixel[1] - orig_window // 2), 0, (orig_window, orig_window))
                    X = X.convert("RGB")
                    image_patch[p][s][row.values[0]] = X
                pbar.update()
    return image_patch

def rotate_and_flip_images(image_patch):
    new_image_patch = collections.defaultdict(dict)
    print("Rotate and flip images")
    with tqdm.tqdm(total=sum(map(len, image_patch.values()))) as pbar:
        for p in image_patch: 
            new_image_patch[p] = {}
            for s in image_patch[p]:
                new_image_patch[p][s] = {}
                for patch_name, img in image_patch[p][s].items():
                    # オリジナル画像を保存
                    new_image_patch[p][s][patch_name] = img

                    # 90度回転して新しいパッチ名を付けて保存
                    new_img = img.rotate(90)
                    new_patch_name = f"{patch_name}_90"
                    new_image_patch[p][s][new_patch_name] = new_img

                    # 180度回転して新しいパッチ名を付けて保存
                    new_img = img.rotate(180)
                    new_patch_name = f"{patch_name}_180"
                    new_image_patch[p][s][new_patch_name] = new_img

                    # 270度回転して新しいパッチ名を付けて保存
                    new_img = img.rotate(270)
                    new_patch_name = f"{patch_name}_270"
                    new_image_patch[p][s][new_patch_name] = new_img

                    # 左右反転して新しいパッチ名を付けて保存
                    new_img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    new_patch_name = f"{patch_name}_flip_horiz"
                    new_image_patch[p][s][new_patch_name] = new_img

                    # 左右反転して90度回転して新しいパッチ名を付けて保存
                    new_img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(90)
                    new_patch_name = f"{patch_name}_flip_horiz_90"
                    new_image_patch[p][s][new_patch_name] = new_img

                    # 左右反転して180度回転して新しいパッチ名を付けて保存
                    new_img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(180)
                    new_patch_name = f"{patch_name}_flip_horiz_180"
                    new_image_patch[p][s][new_patch_name] = new_img

                    # 左右反転して270度回転して新しいパッチ名を付けて保存
                    new_img = img.transpose(Image.FLIP_LEFT_RIGHT).rotate(270)
                    new_patch_name = f"{patch_name}_flip_horiz_270"
                    new_image_patch[p][s][new_patch_name] = new_img
                pbar.update()

    return new_image_patch

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
    image_patch = LoadImageData(patient)
    image_patch = rotate_and_flip_images(image_patch)
    imageToFeature(patient, image_patch)    
main()