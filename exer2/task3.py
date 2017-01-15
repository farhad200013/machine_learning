import numpy as np
fp = open("C:\Users\aliTakin\Desktop\4.92\sgn_41007\locationData\locationData.csv", "r")
import numpy as np
if __name__ == "__main__":
    X = [] # Rows of the file go here
    # We use Python’s with statement.
    # Then we do not have to worry
    # about closing it.
    with open("locationData.csv", "r") as fp:
        # File is iterable, so we can
        # read it directly (instead of
        # using readline).
        for line in fp:              
            # Split the line to numbers:
            values = line.split(" ")
            values = values[😏
                # Cast each item from string to float:
            values = [float(v) for v in values]
                # Append to X
            X.append(values)
    # Now, X is a list of lists. Cast to Numpy array:
    X = np.array(X)
print(X)
print ("All data read.")
print ("Result size is %s" % (str(X.shape)))





#if __name__ == "__main__":
#    X=[]
#    with open("C:\Users\aliTakin\Desktop\4.92\sgn_41007\locationData\locationData.csv", "r") as fp:
#        
#        for line in fp:
#            
#        
#            values = line.split(" ")
#        # Omit the first item
#        # ("S1" or similar):
#            values = values[1:]
#        # Cast each item from
#        # string to float:
#            values = [float(v) for v in values]
#         # Append to X
#            X.append(values)
## Now, X is a list of lists. Cast to
## Numpy array:
#            X = np.array(X)
#print ("All data read.")
#print ("Result size is %s" % (str(X.shape)))
            
# Skip the first line:
