filePath = r'/home/nikki/2012-10-20-coding_data-all.txt'
count = 1
with open(filePath, 'r') as file:
    line = file.readline()
    values = line.split('\t')
    key = values[0]
    for line in file:
        newValues = line.split('\t')
        newKey = newValues[0]
        if newKey != key:
            key = newKey
            count = count + 1
print('count: ', count)