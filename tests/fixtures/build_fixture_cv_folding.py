#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:24:24 2026

@author: kathrin
Creates a fixture dataset for testing cross-validation splitting.

The fixture contains 50 samples distributed across eight classes according
to the class distribution specified in the metadata. Samples are assigned
to six group categories (PAM, E, N, K, C, D). The E groups represent paired
T/P samples, which share the same group identifier.

The generated dataset contains the columns 'code', 'type' and 'group'
and is saved as an Excel file for use in the cross-validation tests.



"""

import pandas as pd
import numpy as np
from pathlib import Path
import random


random.seed(4)
cv_types = ['cocaine', 'MDMA', 'no_controlled_substance', 'amphetamine_type',
       '2C-B_phenethylamine-psychedelics', 'cathinone', 'other', 'ketamine']
cv_group = ["PAM", "E", "N", "K", "C", "D"]
cv_code = ["PAM", "T", "P", "N", "K", "C", "D"]
type_meta = [79, 36, 27, 21, 13, 11, 8, 5] #Distribution of types according top metadatafile. Total = 200 samples
type_list = np.concatenate([[cv_types[i]]*type_meta[i] for i in range(len(type_meta))]).tolist()
random.shuffle(type_list)
def create_group_dist():
    """ Generates random group sizes for the cross-validation fixture.

        Six positive integers are sampled from the range 0 to 29. New values
        are generated until their sum equals 50 and the second value (E group)
        is even. The even group size ensures that all T/P samples can be assigned
        in pairs.
        
        Returns:
            list: Six group sizes corresponding to PAM, E, N, K, C and D.
    """
    rand = random.sample(range(130),6)
    while sum(rand) != 200 or rand[1]%2 != 0 or any(ele == 0 for ele in rand):
        rand = random.sample(range(130),6)
    return rand

num_group = create_group_dist()
#prepare according codes, E must have doubled numbers, cause it is originated from t and p
group_list = [
    cv_group[i] + str(k)
    for i in range(len(cv_group))
    for k in (
        sorted(np.concatenate([
            list(range(1, int(num_group[i]/2) + 1)),
            list(range(1, int(num_group[i]/2) + 1))
        ]).tolist())
        if cv_group[i] == "E"
        else range(1, num_group[i] + 1)
    )
]

#prepare the code list
i = num_group[0]
code_list = []
for group in group_list:
    if group.startswith("E"):
        if i%2 !=0:
            group = group.replace("E","T")
        else:
            group = group.replace("E","P")
        i+=1
    code_list.append(group)

dic = {"code" : code_list,
       "type" : type_list,
       "group" : group_list }
df = pd.DataFrame(dic)


save_path = Path(__file__).parent / "cv_folding_fixture.xlsx"
df.to_excel(save_path, index=False)
print(f"Fixture written to {save_path}")





