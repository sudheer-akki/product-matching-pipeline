#!/bin/bash

mongoimport --db='products' \
            --collection='metadata' \
            --file='/tmp/metadata.json' \
            --jsonArray \
            --username='admin' \
            --password='admin123' \
            --authenticationDatabase=admin
