#!/bin/sh

mkdir -p models 
cd models

######################
# slo models from https://www.clarin.si/repository/xmlui/handle/11356/1736
######################

# en sl model
wget "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736/ensl_GEN_nemo-1.2.6.tar.zst?sequence=2&isAllowed=y" -O /tmp/ensl_GEN_nemo-1.2.6.tar.zst
zstd -d /tmp/ensl_GEN_nemo-1.2.6.tar.zst && rm /tmp/ensl_GEN_nemo-1.2.6.tar.zst
tar -xf /tmp/ensl_GEN_nemo-1.2.6.tar && rm /tmp/ensl_GEN_nemo-1.2.6.tar

# sl en model
wget "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736/slen_GEN_nemo-1.2.6.tar.zst?sequence=1&isAllowed=y" -O /tmp/slen_GEN_nemo-1.2.6.tar.zst
zstd -d /tmp/slen_GEN_nemo-1.2.6.tar.zst && rm /tmp/slen_GEN_nemo-1.2.6.tar.zst
tar -xf /tmp/slen_GEN_nemo-1.2.6.tar && rm /tmp/slen_GEN_nemo-1.2.6.tar

######################
# other models from: https://catalog.ngc.nvidia.com/models?filters=&orderBy=scoreDESC&query=nemo+nmt
######################

# es en
mkdir -p esen
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_es_en_transformer24x6/versions/1.5/zip -O /tmp/nmt_es_en_transformer24x6_1.5.zip
unzip /tmp/nmt_es_en_transformer24x6_1.5.zip -d esen && rm /tmp/nmt_es_en_transformer24x6_1.5.zip
cat >esen/model.info <<EOL
language_pair: esen
domain: DEFAULT
version: nemo-1.0.0
info:
  framework: nemo:es_en_24x6
EOL

# en es
mkdir -p enes
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_es_transformer24x6/versions/1.5/zip -O /tmp/nmt_en_es_transformer24x6_1.5.zip
unzip /tmp/nmt_en_es_transformer24x6_1.5.zip -d enes && rm /tmp/nmt_en_es_transformer24x6_1.5.zip
cat >enes/model.info <<EOL
language_pair: enes
domain: DEFAULT
version: nemo-1.0.0
info:
  framework: nemo:en_es_24x6
EOL
