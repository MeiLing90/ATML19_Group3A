import os
import glob
import os
import glob
fileList = glob.glob('./data/**/**/depth*', recursive=True)
for filePath in fileList:
    os.remove(filePath)

fileList = glob.glob('./data/**/**/*.png', recursive=True)
i=0
for filePath in fileList:
    i = i+1
    parts = (filePath.split('/'))[:-1]
    if(parts[2] in ["A","B","C","D"]):
        path = parts[0]+"/"+parts[1]+"/A/"+parts[3]+"/"
        os.rename(filePath,path+"train_"+str(i).zfill(6)+".png")