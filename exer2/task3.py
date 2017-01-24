import numpy as np
fp = open(r"C:\Users\aliTakin\Desktop\4.92\sgn_41007\locationData\locationData.csv", "r")
if __name__ == "__main__":
    X = [] # Rows of the file go here
    # about closing it.
    with open(r"C:\Users\aliTakin\Desktop\4.92\sgn_41007\locationData\locationData.csv", "r") as fp:
        # File is iterable, so we can
        # read it directly (instead of
        # using readline).
        for line in fp:              
            # Split the line to numbers:
            values = line.split(' ')
            
                # Cast each item from string to float:
            values = [float(v) for v in values]
                # Append to X
            X.append(values)
    X = np.array(X)
    x2=np.loadtxt(r"C:\Users\aliTakin\Desktop\4.92\sgn_41007\locationData\locationData.csv")
print(np.all(X==x2))

print ("size is %s" % (str(X.shape)))





