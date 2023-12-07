import re
unfiltered = [line.rstrip('\n') for line in open("davidson.txt")]
pattern = re.compile(r'^\d+,')  # ^\d+ matches one or more digits at the beginning of a line, followed by a comma
filtered = [line for line in unfiltered if line and pattern.match(line)]

print(len(filtered))
dv = ['id,tweet,label']
with open(r'dv_train.txt', 'w') as fp:
    for line in filtered: 
        parts = line.split(",")
        fp.write("%s\n" % (f"{parts[0]},\"{', '.join(parts[6:])}\",{parts[5]}"))