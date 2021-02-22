# First script does data cleaning/label encoding, while the second one
# creates a validation scheme.
python3 -m src.prepare_data
python3 -m src.create_folds