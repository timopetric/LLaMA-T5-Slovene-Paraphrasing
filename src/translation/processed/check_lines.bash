#!/bin/bash

# exploratory analysis script to double check that the sentences are
# aligned between the 3 files (orig-sl, orig-en, tran-sl)

# Set the number of sentences to print
NUM_SENTENCES=100

# Get the file names from command line arguments
FILE_1=$1
FILE_2=$2
FILE_3=$3

# Get the number of lines in the files
NUM_LINES=$(wc -l < $FILE_1)

# Generate a list of random line numbers
LINE_NUMBERS=$(shuf -i 1-$NUM_LINES -n $NUM_SENTENCES)

# Loop through the line numbers and print the corresponding sentences
for LINE_NUMBER in $LINE_NUMBERS; do
    # Get the sentence from each file
    SENTENCE_1=$(sed -n "${LINE_NUMBER}p" $FILE_1)
    SENTENCE_2=$(sed -n "${LINE_NUMBER}p" $FILE_2)
    SENTENCE_3=$(sed -n "${LINE_NUMBER}p" $FILE_3)

    # Print the sentences
    echo "#######################################"
    echo "---------- $FILE_1 -----------"
    echo "$SENTENCE_1"
    echo "---------- $FILE_2 -----------"
    echo "$SENTENCE_2"
    echo "---------- $FILE_3 -----------"
    echo "$SENTENCE_3"
done
