# import re
# unfiltered = [line.rstrip('\n') for line in open("davidson.txt")]
# pattern = re.compile(r'^\d+,')  # ^\d+ matches one or more digits at the beginning of a line, followed by a comma
# filtered = [line for line in unfiltered if line and pattern.match(line)]

# print(len(filtered))
# dv = ['id,tweet,label']
# with open(r'dv_train.txt', 'w') as fp:
#     for line in filtered: 
#         parts = line.split(",")
#         tweet = ', '.join(parts[6:]).replace("\"", "")
#         fp.write("%s\n" % (f"{parts[0]},\"{tweet}\",{parts[5]}"))

import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv('dv.csv')

# Split the dataset into training and testing sets
train, test = train_test_split(df, test_size=0.1, random_state=42)

# Save the training and testing sets into separate CSV files
train.to_csv('dv_train.csv', index=False)
test.to_csv('dv_test.csv', index=False)