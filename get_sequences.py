import pickle
import pandas as pd

#binding_pkl = open("binding.pkl","rb")
#khsrp_binding  = pd.read_pickle(binding_pkl,compression=None)

khsrp_binding = pd.read_hdf('binding_FXR1.hdf','data')

#not_binding_pkl= open("not_binding.pkl","rb")
#not_khsrp_binding = pd.read_pickle(not_binding_pkl,compression=None)

not_khsrp_binding = pd.read_hdf('not_binding_FXR1.hdf','data')

genome = pickle.load(open("hg38.pkl","rb"))

f1 = open("pos_sequences_FXR1.txt", "a")
f2 = open("neg_sequences_FXR1.txt","a")
context_length = 100

prev_end = 0
prev_chrom = ""
print(not_khsrp_binding.shape[0])
print(khsrp_binding.shape[0])

count = 0

complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

for i in range(0, not_khsrp_binding.shape[0]):
    count += 1
    if i < khsrp_binding.shape[0]:
      row = khsrp_binding.iloc[i]
      midpoint = int(.5 * (int(row.start) + int(row.end)))
      seq = genome[row.chrom][ midpoint - context_length//2:midpoint + context_length//2]
      if row.direction == "-":
        seq = "".join(complement.get(base, base) for base in reversed(seq))
      f1.write(seq+"\n")
    row = not_khsrp_binding.iloc[i]
    midpoint = int(.5 * (int(row.start) + int(row.end)))
    seq = genome[row.chrom][ midpoint - context_length//2:midpoint + context_length//2]
    f2.write(seq+"\n")

    prev_chrom = row.chrom
    prev_end = row.end
print(count)
f1.close()
f2.close()
