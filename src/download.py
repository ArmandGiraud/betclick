"""utilities to download and extract data """
import os
import urllib
import zipfile as zf


FILEID = "1THuNxeDr2Qh4DcVrVKLJeHB3QkZtu7uV"
URL = "https://docs.google.com/uc?export=download&id={}".format(FILEID)


def download(url, save_path):
    "download the google drive archive"
    _, _ = urllib.request.urlretrieve(url=url, filename=save_path)

def download_extract(dir_path):
    """download the data if the data file is not present
    in the directory given as argument and unzip it in the same folder"""
    save_path = os.path.join(dir_path, "betclick.zip")
    dest_path = os.path.join(dir_path, "betclic_datascience_test_churn.csv")
    if not os.path.exists(dest_path):
        print('file not found in {} downloading...'.format(dest_path))
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)

        download(URL, save_path)

        with zf.ZipFile(save_path, 'r') as zip_ref:
            password = input("enter zip password: ")
            zip_ref.setpassword(str.encode(password))

            print("extracting dataset in {}".format(dir_path))
            zip_ref.extractall(dir_path)
