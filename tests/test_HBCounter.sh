python ../HBCounter.py . -n 2 -t output/topologies/conntopology_0.pdb

python ../HBCounter.py sim1 sim2 -n 2 -t output/topologies/conntopology_0.pdb --include_rejected_steps --report_name mod_report -o Hbs.csv --alternative_output_path my_custom_path

python ../HBCounter.py sim1 sim2 -n 2 -t output/topologies/conntopology_0.pdb
