import matplotlib.pyplot as plt
import RNA


pos_sequences = open('pos_sequences_FXR1.txt','r')
output_file = open('pos_dotbracket_FXR1.txt','w')

count = 0

for seq in pos_sequences:
	# The RNA sequence
	# compute minimum free energy (MFE) and corresponding structure
	(ss, mfe) = RNA.fold(seq.rstrip())
	# print output
	output_file.write(ss+"\n")
	if count % 1000 == 0:
		print(count)
	count += 1

output_file.close()


neg_sequences = open('neg_sequences_FXR1.txt','r')
output_file = open('neg_dotbracket_FXR1.txt','w')

count = 0

for seq in neg_sequences:
        # The RNA sequence
        # compute minimum free energy (MFE) and corresponding structure
        (ss, mfe) = RNA.fold(seq.rstrip())
        # print output
        output_file.write(ss+"\n")
        if count % 1000 == 0:
                print(count)
        count += 1

output_file.close()
