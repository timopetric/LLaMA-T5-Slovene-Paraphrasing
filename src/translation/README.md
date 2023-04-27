# Translation 

We translate sentences using the Slovene NMT.

To setup make sure you have the following directories available here (can be symlinks):
+ `containers` - used for `nmt.sif` singularity container with the needed environment (image definition in `Singularity.def`)
+ `models`- used for NeMO translation models (from [Neural Machine Translation model for Slovene-English language pair RSDO-DS4-NMT 1.2.6](https://www.clarin.si/repository/xmlui/handle/11356/1736)).
    + look at `models_setup.sh` to download the wanted models
+ `opus` - corpuses in subfolders, each with 3 files that you then define in constants in `read_sentences.py`. This are the files that will be translated.
+ `logs` - a directory that will hold the logs
