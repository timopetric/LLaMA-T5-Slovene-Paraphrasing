#!/bin/sh

mkdir -p models 
cd models

# en sl model
wget "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736/ensl_GEN_nemo-1.2.6.tar.zst?sequence=2&isAllowed=y" -O ensl_GEN_nemo-1.2.6.tar.zst
zstd -d ensl_GEN_nemo-1.2.6.tar.zst && rm ensl_GEN_nemo-1.2.6.tar.zst
tar -xf ensl_GEN_nemo-1.2.6.tar && rm ensl_GEN_nemo-1.2.6.tar

# sl en model
wget "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736/slen_GEN_nemo-1.2.6.tar.zst?sequence=1&isAllowed=y" -O slen_GEN_nemo-1.2.6.tar.zst
zstd -d slen_GEN_nemo-1.2.6.tar.zst && rm slen_GEN_nemo-1.2.6.tar.zst
tar -xf slen_GEN_nemo-1.2.6.tar && rm slen_GEN_nemo-1.2.6.tar
