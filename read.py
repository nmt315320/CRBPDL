import os, time, sys, re
import pandas as pd
fw1 = open("./Datasets/linRNA-RBP/"+protein+"/negative.txt", "w")
fw0 = open("./Datasets/linRNA-RBP/"+protein+"/positive.txt", "w")
file = open('./Datasets/linRNA-RBP/1AGO1/sequences.fa')
sum = 0
l1 = []
l0 = []
with open("./Datasets/linRNA-RBP/2AGO2/sequences.fa", "r") as f:
    lines = f.readlines()
    i=0
    key = ["class:1","class:0"]
    for line in lines :
        line = line.rstrip()
        if key[0] in line:
            name = line[1:]
            l1.append(lines[i])
            l1.append(lines[i+1])
            l1.append(lines[i+2])
            i=i+3
        else:
            if key[1] in line:
                l0.append(lines[i])
                l0.append(lines[i + 1])
                l0.append(lines[i + 2])
                i=i+3
fw0.writelines(l0)
fw1.writelines(l1)
f.close()
fw0.close()
fw1.close()
