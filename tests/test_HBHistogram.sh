python ../HBHistogram.py hbonds.out -m count &
python ../HBHistogram.py hbonds.out -m relative_frequency &
python ../HBHistogram.py hbonds.out -m frequent_interactions &
python ../HBHistogram.py hbonds.out -m mean_energies --models_to_ignore 0
