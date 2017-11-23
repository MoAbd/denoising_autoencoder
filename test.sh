#!/bin/bash

#for i in 0.5
#do
    #/usr/bin/python3.5 run_autoencoder.py --corr_type masking --corr_frac $i --model_name model_masking_frac_$i > model_masking_frac_$i.txt
#done

for i in 0.3 0.4 0.5
do
    /usr/bin/python3.5 run_autoencoder.py --corr_type salt_and_pepper --corr_frac $i --model_name model_sap_frac_$i > model_sap_frac_$i.txt
done

for i in 0.1 0.2 0.3 0.4 0.5
do
    /usr/bin/python3.5 run_autoencoder.py --corr_type gaussian --corr_frac $i --model_name model_gaussian_frac_$i > model_gaussian_frac_$i.txt
done




