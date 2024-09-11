#!/bin/bash

# Run the docker container with the Widoco image
# The container will generate the documentation for the ontology
# The documentation will be saved in the data/target/generated-doc folder
# The ontology file is located in the data/ontologies folder
# The ontology file is wirr_project3.rdf

sudo docker run -ti --rm \
    -v `pwd`/data/ontologies:/usr/local/widoco/in:Z \
    -v `pwd`/data/target/generated-doc:/usr/local/widoco/out:Z \
    ghcr.io/dgarijo/widoco:v1.4.25 \
    -ontFile in/wirr_project3.rdf \
    -outFolder out \
    -rewriteAll \
    -webVowl
