#!/bin/bash

echo "Running fetch_dataset script"
sh scripts/fetch_dataset.sh

echo "\n\nRunning create_indexes script...\n"
sh scripts/create_indexes.sh

echo "\n\nDONE!"
