#!/bin/sh

seed=42
collection_mode='training'  # validation

python3 automatic_dataset_collection.py --seed "$seed" \
                                        --loop \
                                        --collection_mode "$collection_mode" \

exit 0