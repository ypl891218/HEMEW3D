#!/bin/bash

# velocity103500-103599.zip

base_idx=100000
total_data=30000
zip_size=100
total_zips=$((total_data/zip_size))

mkdir velocity

for (( i=0; i < $total_zips; ++i )); do
	start_idx=$((base_idx+zip_size*i))
	end_idx=$((start_idx+zip_size-1))
	echo $start_idx
	echo $end_idx
	unzip -o velocity${start_idx}-${end_idx}.zip
	mv velocity${start_idx}-${end_idx}/* ./velocity/
	rm -r velocity${start_idx}-${end_idx}
done
