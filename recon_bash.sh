# loop over P001 to P010
for i in $(seq -f "%03g" 1 10); do
    echo "Processing Patient P${i}"
    # python CMRxRecon_gt_sense.py --file_name /csiNAS3/mridata/CMRxRecon/TrainingSet/P${i}/cine_lax_ks.mat --out_file cine_lax_${i}.pt #&
    python CMRxRecon_gt_sense.py --file_name /csiNAS3/mridata/CMRxRecon/TrainingSet/P${i}/cine_sax_ks.mat --out_file cine_sax_${i}.pt #&
    # wait  # Wait for both commands to complete before moving to next patient
done

