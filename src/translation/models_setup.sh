#!/bin/sh

# script that downloads different Nemo models for machine translation and puts them in models dir
# models dir can be a simbolic link pointing somewhere or it can be a folder (default)

echo "THIS CODE IS NOT PARTICULARY TESTED AND MAYBE DOES NOT WORK FULLY. PROCEDE WITH CAUTION" && \
  echo "" && \
  exit 0

MODELS_DIR_NAME="models_test"
DIRECTORY_LINK="/d/hpc/projects/FRI/tp1859/nlp_project8/$MODELS_DIR_NAME"


[ -d $DIRECTORY_LINK ] && echo -e "Directory $DIRECTORY_LINK already exists." && exit 1

# create models dir here
# mkdir -p models

# or create a dir somewhere and create a link to it (optional instead of just mkdir)
mkdir -p $DIRECTORY_LINK
ln -s $DIRECTORY_LINK $MODELS_DIR_NAME

# move inside the created dir
cd $MODELS_DIR_NAME


######################
# slo models from https://www.clarin.si/repository/xmlui/handle/11356/1736
######################

# en sl model
wget "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736/ensl_GEN_nemo-1.2.6.tar.zst?sequence=2&isAllowed=y" -O "ensl_GEN_nemo-1.2.6.tar.zst" && \
    zstd -d "ensl_GEN_nemo-1.2.6.tar.zst" && rm "ensl_GEN_nemo-1.2.6.tar.zst" && \
    tar -xf "ensl_GEN_nemo-1.2.6.tar" && rm "ensl_GEN_nemo-1.2.6.tar" && \
    echo -e "\n[+] ensl_GEN_nemo successfully downloaded and unzipped\n" || echo -e "\n[-] ensl_GEN_nemo error\n"


# sl en model
wget "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1736/slen_GEN_nemo-1.2.6.tar.zst?sequence=1&isAllowed=y" -O "slen_GEN_nemo-1.2.6.tar.zst" && \
    zstd -d "slen_GEN_nemo-1.2.6.tar.zst" && rm "slen_GEN_nemo-1.2.6.tar.zst" && \
    tar -xf "slen_GEN_nemo-1.2.6.tar" && rm "slen_GEN_nemo-1.2.6.tar" && \
    echo -e "\n[+] slen_GEN_nemo successfully downloaded and unzipped\n" || echo -e "\n[-] slen_GEN_nemo error\n"

######################
# other models from: https://catalog.ngc.nvidia.com/models?filters=&orderBy=scoreDESC&query=nemo+nmt
######################

# es en
[ -d esen ] && echo -e "Directory esen already exists." && exit 1
mkdir -p esen && \
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_es_en_transformer24x6/versions/1.5/zip -O nmt_es_en_transformer24x6_1.5.zip && \
  unzip nmt_es_en_transformer24x6_1.5.zip -d esen && rm nmt_es_en_transformer24x6_1.5.zip && \
cat >esen/model.info <<EOL
language_pair: esen
domain: DEFAULT
version: nemo-1.0.0
info:
  framework: nemo:es_en_24x6
EOL && \
  echo -e "\n[+] nmt_es_en_transformer24x6 successfully downloaded and unzipped\n" || echo -e "\n[-] nmt_es_en_transformer24x6 error\n"


# en es
[ -d enes ] && echo -e "Directory enes already exists." && exit 1
mkdir -p enes && \
  wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/nemo/nmt_en_es_transformer24x6/versions/1.5/zip -O nmt_en_es_transformer24x6_1.5.zip && \
  unzip nmt_en_es_transformer24x6_1.5.zip -d enes && rm nmt_en_es_transformer24x6_1.5.zip && \
  cat >enes/model.info <<EOL
language_pair: enes
domain: DEFAULT
version: nemo-1.0.0
info:
  framework: nemo:en_es_24x6
EOL && \
  echo -e "\n[+] nmt_en_es_transformer24x6 successfully downloaded and unzipped\n" || echo -e "\n[-] nmt_en_es_transformer24x6 error\n"
