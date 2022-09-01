# temporal_networks
Temporal Networks Visualization using igraph on R and python

These scripts show how to produce dynamical visualizations of a temporal network in R and python with the amazing package igraph. 


# R
Use the new version of the script (preferentialattachment_new.r) if you are using the last version of the igraph package (1.0)

You can find more information about the visualization and the corresponding script here: http://estebanmoro.org/2015/12/temporal-networks-with-r-and-igraph-updated/


# Python
Ensure you active venv/ and please install the following (inside a venv)

```
python3 -m venv venv
source venv/bin/activate
pip install igraph
pip install pandas
pip install pycairo
pip install matplotlib
pip install pycountry
pip install progress
```

## Running it
Look at the script generate-anim.sh. It encapsulates the commands. Please make sure you run it inside the venv

```
./generate-anim.sh
```

## Results
check animation/ folder and delete it between runs. Note, there is a small dataset for testing as well as a the larger one
