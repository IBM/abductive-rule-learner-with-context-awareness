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

In the latest version of the code we migrated from the original numpy-based dataset to a JSON-based following the approach of [Hu et al.](https://github.com/hxiaoyang/lm-raven).
To convert the original dataset to the JSON files required by the new dataloader, use the script provided [here](https://github.com/IBM/raven-large-language-models/blob/main/src/datasets/generation/iraven_task.py).
