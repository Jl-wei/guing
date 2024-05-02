import os
import argparse

if __name__ == "__main__":
    file = open('nohup.out', 'r')
    lines = file.readlines()
    
    precision = 0
    recall = 0
    f1 = 0
    count = 0
    for line in lines:
        if line.startswith("weighted avg"):
            count += 1
            result = line.split("     ")
            precision += float(result[1])
            recall += float(result[2])
            f1 += float(result[3])

    print("precision", precision/count)
    print("recall", recall/count)
    print("f1", f1/count)
