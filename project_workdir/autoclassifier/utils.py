# -*- coding: utf-8 -*-
from .config import preproc_dir_var 


def create_dir_link(data_path = preproc_dir_var,
                        filename = 'text.csv'):
    return str(data_path/filename)


def write_tocsv(df, data_path=None, fname = 'file.csv'):
    if not data_path:
        fpath = create_datadir_link(filename = fname)
    else:
        fpath = create_datadir_link(data_path = data_path, filename = fname)
    try:
        df.to_csv(fpath)
    except:
        print('WARNING: Could not save file, check if dataframe was created properly or path is right.')

