# -*- coding: utf-8 -*-
from .config import preproc_dir_var 


def create_dir_link(data_path = preproc_dir_var,
                        filename = 'text.csv'):
    """
    Used to create a str that contains path to file using a Path object.
    
    Parameters
    ----------
    data_path: Path object, optional
        Path to file, passed as a Path object using the Path library.
        Default data_dir_var taken from config file
    
    filename: str, optional
        Name of the file
        Default 'text.csv'
    
    Returns
    -------
    str
        Path to filename
    """  
    return str(data_path/filename)


def write_tocsv(df, data_path=None, fname = 'file.csv'):
    """
    Writes a DataFrame to csv using the paths provided in the config files.
    
    Parameters
    ----------
    df: Pandas.DataFrame
    
    data_path: Path object, optional
        The path to the file generated using the Path library
        Default None
    
    filename: str, optional
        Filename for the csv file.
        Defailt file.sav
       
    Returns
    -------
    None
             
    """
    if not data_path:
        fpath = create_dir_link(filename = fname)
    else:
        fpath = create_dir_link(data_path = data_path, filename = fname)
    try:
        df.to_csv(fpath)
    except:
        print('WARNING: Could not save file, check if dataframe was created properly or path is right.')

