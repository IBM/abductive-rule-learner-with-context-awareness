## Get the Data
Generate the I-RAVEN dataset with the instructions proveded [here](https://github.com/husheng12345/SRAN) and save it in this folder.

```bash
git clone https://github.com/husheng12345/SRAN
pip2 install --user -r SRAN/I-RAVEN/requirements.txt
python2 SRAN/I-RAVEN/main.py --save-dir .
```

## Prepare the Data

Run the rule preprocessing script:
```bash
python arlc/utils/raven/extraction.py --data_path data
```
