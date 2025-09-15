#%%
import sys
import os
import functools
import pandas as pd
#%%

def csvWriter(prefix=None):
    """
    A decorator to read data/write data to a directory as csv
    instead of parsing it every time.
    To use add the kwarg:
        csvDir - the directory to read/write csv
        force - bool to force parsing data and write new csv

    Parameters
    ----------
    func : function
        function to be wrapped
    Returns
    -------
    function
        the wrapper for the func argument
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            dir = args[0]
            force = False
            if "csvDir" in kwargs:
                csvDir = kwargs.pop("csvDir")
                if "force" in kwargs:
                    force = kwargs.pop("force")
            else:
                return func(*args, **kwargs)

            ret = None
            csvName = None
            if csvDir:
                tempDir = dir.replace("/", "_")
                tempDir = tempDir.replace(".", "_")
                if prefix is not None:
                    tempDir = prefix + "_" + tempDir
                csvName = csvDir + "/" + tempDir + ".csv"
                if csvDir is not None and not force:
                    try:
                        ret = pd.read_csv(csvName, index_col=0)
                    except:
                        force = True

            if ret is None or force:
                ret = func(*args, **kwargs)
                if csvName is not None:
                    print("Writing", csvName)
                    ret.to_csv(csvName)
            else:
                print("Reading", csvName)
            return ret

        return wrapper
    return decorator

def getFiles(dir, filter=None, singleLevel=False):
    """
    This grabs all the files in a directory.  The filter is used
    to select files that match a sub-string.  If no filters is
    supplied, all files in the directory will be return.

    Parameters
    ----------
    dir : str
        Directory to get files from
    filter : str, optional
        Sub-strings to look for in filename

    Returns
    -------
    list
        sorted list of filenames
    """
    ret = []
    for root, dirs, files in os.walk(dir):
        for f in files:
            if filter is None or filter in f:
                file = os.path.join(root, f)
                ret.append(file)
        if singleLevel:
            break
    ret.sort()
    return ret

def parse(filename):
    # folder = filename.split("/")[-2]
    # whichInt = int(folder.split("_")[-1]) - 1
    # lastPart = filename.split("/")[-1]
    # base = lastPart[:-4]
    # parts = base.split("_")
    # ranks = int(parts[1])
    # threads = int(parts[2])
    # rep = int(parts[3])

    folder = filename.split("/")[-2]
    which = folder.split("_")[-1]
    delay = int(folder.split("d")[0])
    lastPart = filename.split("/")[-1]
    base = lastPart[:-4]
    parts = base.split("_")
    # ranks = int(parts[1])
    threads = int(parts[2])
    rep = int(parts[3])
    

    
    mapper = ['Node1', 'Node2', 'Both', 'Node1_Base', 'Node2_Base']

    ret = []
    with open(filename) as f:
        lines = f.readlines()
        # passed = len([lines for line in lines if "PASSED" in line]) == ranks
        lines = [line for line in lines if "Rank: " in line and "TID: " in line]
        for line in lines:
            # print(line)
            line_parts = line.split()
            rank = int(line_parts[1])
            tid = int(line_parts[3])
            count = int(line_parts[5])
            time = float(line_parts[7])
            # for i in range(count):
            ret.append({"which":which, "delay":delay, "threads":threads, "reps":rep, "rank":rank, "tid":tid, "count":count, "time":time/count, "totalTime": time, "filename": filename})
    return ret

@csvWriter()
def getDfFromDir(dir):
    files = getFiles(dir, ".txt")
    temp = []
    for f in files:
        print(f)
        # try:
        temp += parse(f)
        # except:
            # print("Failed", f)
    return pd.DataFrame(temp)

#%%
if __name__ == "__main__":
    df = getDfFromDir(sys.argv[1], csvDir=sys.argv[2], force=True)
    print(df.head())
        
#%%