import os
import glob
import os
import glob
fileList = glob.glob('/var/tmp/jiyoung/data/**/**/depth*', recursive=True)
for filePath in fileList:
    os.remove(filePath)

fileList = glob.glob('/var/tmp/jiyoung/data/**/**/*.png', recursive=True)
i=0
for filePath in fileList:
    i = i+1
    parts = (filePath.split('/'))[:-1]
    if(parts[5] in ["A","B","C","D"]):
        path = "/var/tmp/jiyoung/data/A/"+parts[6]+"/"
        if i==1:
            print(path+"train_"+str(i).zfill(6)+".png")
        os.rename(filePath,path+"train_"+str(i).zfill(6)+".png")
