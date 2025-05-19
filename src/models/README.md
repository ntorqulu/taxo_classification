### Steps to execute the base model

This is just a very simple example to have a working base model and a starting 
point for further development.


1. Install the dependencies from `requirements.txt`.
2. In `models/main.py`, set the value of `HPARAMS['taxo_path']` to the full path of your CSV file.
It’s important that this path points to the CSV file.
3. Run `models/main.py`.

> ⚠️ **Warning:** The first execution may take around an hour due to k-mer encoding.  
>  If you have a Parquet file, it must be in the same directory and have the same name but with a different extension.
> For instance: `database.csv` and `database.parquet`.

