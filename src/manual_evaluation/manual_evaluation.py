import random
import datetime
import os

task = ["grammar", "lexical_divergence", "meaning_preservation", "fluency"]

def get_random_line(file1, file2):
    # Read a line from each file
    line1 = file1.readline()
    line2 = file2.readline()

    if random.choice([True, False]):
        return line1, line2, 1  # line1 is first
    else:
        return line2, line1, 2  # line2 is first
    

def get_user_scores(sentence):
    scores = []
    
    print("Sentence: ", sentence)
    for i in range(4):
        score = input(task[i] + "(1-5): ")
        scores.append(score)
    return scores

# file1_path = "/home/domen/Documents/Faks/NLP/Manual Evaluation/mocacu/best_paraphrases_mocacu_t5.txt"
# file2_path = "/home/domen/Documents/Faks/NLP/Manual Evaluation/mocacu/4MaCoCu-1000_llamapara_en_to_sl_tran_from3.out"

file1_path = "mocacu/best_paraphrases_mocacu_t5.txt"
file2_path = "mocacu/4MaCoCu-1000_llamapara_en_to_sl_tran_from3.out"

# file_original_path = "/home/domen/Documents/Faks/NLP/Manual Evaluation/mocacu/1mocacu_200x_orig_sl.txt"
file_original_path = "mocacu/1mocacu_200x_orig_sl.txt"

scores_folder_path = "scores"

current_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")

# file1_scores_path = f"/home/domen/Documents/Faks/NLP/Manual Evaluation/scores/scores_t5_{current_datetime}.txt"
# file2_scores_path = f"/home/domen/Documents/Faks/NLP/Manual Evaluation/scores/scores_vicuna_{current_datetime}.txt"

file1_scores_path = os.path.join(scores_folder_path, f"scores_t5_{current_datetime}.txt")
file2_scores_path = os.path.join(scores_folder_path, f"scores_vicuna_{current_datetime}.txt")

if not os.path.exists(scores_folder_path):
    os.makedirs(scores_folder_path)

#     open(file_original_path, "r") as file_original, \

with open(file1_path, "r") as file1, open(file2_path, "r") as file2, \
     open(file1_scores_path, "w") as file1_scores, \
     open(file_original_path, "r") as file_original, \
     open(file2_scores_path, "w") as file2_scores:
    
    file1_scores.write(",".join(task) + "\n")
    file2_scores.write(",".join(task) + "\n")

    user = input("Enter a, b or c to start the evaluation: ")

    if (user == "a"):
        start_line = 0  # Define the starting line
        end_line = 30  # Define the ending line
    elif (user == "b"):
        start_line = 30
        end_line = 60
    elif (user == "c"):
        start_line = 60
        end_line = 90
    

    for _ in range(start_line):
        file_original.readline()
        file1.readline()
        file2.readline()


    #for line1, line2, line_original in zip(file1, file2, file_original):
    #for line1, line2 in zip(file1, file2):
    for _ in range(end_line - start_line):
        # Get a random line from the files
        #
        line_original = file_original.readline()
        line1 = file1.readline()
        line2 = file2.readline()
        print("Original sentence: ", line_original)
        print("-----------------------------------")
        
        if random.choice([True, False]):
            sentence1 = line1
            sentence2 = line2
            file_indicator = 1
        else:
            sentence1 = line2
            sentence2 = line1
            file_indicator = 2
        
        scores1 = get_user_scores(sentence1)
        scores2 = get_user_scores(sentence2)

        
        if file_indicator == 1:
            file1_scores.write( ",".join(scores1) + "\n")
            file2_scores.write(",".join(scores2) + "\n")
        else:
            file1_scores.write(",".join(scores2) + "\n")
            file2_scores.write(",".join(scores1) + "\n")

