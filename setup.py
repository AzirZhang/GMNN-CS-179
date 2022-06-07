import os

os.mkdir("semisupervised/data/citeseer/params")
os.mkdir("semisupervised/data/cora/params")
os.mkdir("semisupervised/data/pubmed/params")
for i in range(100):
    os.mkdir(f"semisupervised/data/citeseer/params/direc{i}")
    os.mkdir(f"semisupervised/data/cora/params/direc{i}")
    os.mkdir(f"semisupervised/data/pubmed/params/direc{i}")