
.PHONY: pred run

all: clean run pred
pred: pred/status.csv
run: save/output.hdf5


pred/status.csv: save/output.hdf5
	rm -rf pred
	mkdir -p pred
	py_smurff --save-freq -1 --save-prefix pred --save-name save/output.hdf5 --row-features feat_0_0.mtx --save-extension .csv

save/output.hdf5:
	rm -rf save
	mkdir -p save
	smurff --ini macau.ini
	
clean:
	rm -fr save pred
	mkdir -p pred save

