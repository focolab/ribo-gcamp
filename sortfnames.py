import os

def sortfn(path):
    listfiles = list(filter(lambda f:f.endswith('ome.tif'), os.listdir(path)))
    fnum = {}
    for s in listfiles:
        s = os.path.join(path,s)
        # fnum[s]=s.split('.ome.tif')[0]
        # fnum[s]=fnum[s].split('Default')[1]
        # if fnum[s] == '':
        #     fnum[s]="00"
        # if len(fnum[s])==2:
        #     fnum[s]="0"+fnum[s][1]
        # if len(fnum[s])>=3:
        #     fnum[s]=fnum[s][1:]
    fn = sorted(fnum.keys(),key=fnum.get)
    return fn