### Steps to execute the base model

This is just a very simple example to have a working base model and a starting 
point for further development.


1. Install the dependencies from `requirements.txt`.
2. In `models/main.py`, set the value of `HPARAMS['taxo_path']` to the full path of your 
CSV file.
3. Run `models/main.py`.

> ⚠️ **Warning:** The first execution may take around an hour due to k-mer loading.  
> I’ll update it ASAP to use a serialized version and avoid this delay.

