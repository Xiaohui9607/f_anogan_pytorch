
import os
import re
import numpy as np

supername = "cifar10"
identifiers = ["_0","_1","_2","_3","_4","_5","_6","_7","_8","_9"]
path="."

for idn_i in identifiers:
    files = [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and re.search(supername+'_\d'+idn_i, f))]
    AUC_list = []
    for fname_i in files:
        with open(fname_i) as f:
            #Assuming the most recent trial is the last line
            for line in f:
                pass
            last_line = line

        match = re.search('ROC_AUC (\d+\.\d+)', line)
        if match:
            AUC_list.append(float(match.group(1)))

    print(idn_i,np.mean(AUC_list))
