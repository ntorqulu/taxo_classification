# taxo_classification

## Introduction
With the advent of the new DNA sequencing thecnologies, methodologies based on such thecniques have unveiled a bast field of applications to infer the Biodiversity of the ecosystems to monitorize such environments and their habitats. Among these thecniques DNA metabarcoding has emerged as a powerfull tool for biomonitoring. This thechnique relies on sequencing a targuet region of the genome of thousands of individuals at the same time and their latter assignment to the different taxa. However, one of the main constrains of the method is this taxonomic assignment due to the missing information in the databases and the amount of resources required to compute such analysis. While the latter problem has been assessed by using bigger computing services and optimizing the algorithms, to be able to run such models in the field withouth internet connection is still a challange. 

Here we propose Deep Learning models as a potential solution for such problem. While the training process is resource demanding, deppending on the architecture of the module, the predictions can be potentially manageable. In this project we explore the following bottlenecks of the implementation of this thecnology in the field of Biomonitoring:

### DNA codification:
We explored three different approaches:
- kmerisation
- 4xN matrix
- One-hot-encoding

### Model architechture
- Fully connected
- CNN

### Category Balance for training

### Hyperparameter optimisation

## Methodology
To obtained the results and to replicate them here we present the different steps followed for the analysis

### Obtain Database
To obtain the database we used the sequences from [mkCOInr](https://github.com/meglecz/mkCOInr) as described in [NJORDR-MJOLNIR3](https://github.com/adriantich/NJORDR-MJOLNIR3) and can be downloaded from google drive: [NJORDR_sequences](https://drive.google.com/file/d/1YU_jIRIm9rpEC4okD5xh2qr3EnGBg3i8/view?usp=sharing), [names.dmp](https://drive.google.com/file/d/1WrRHX5Mf23ijg03K5YNaAx3dtzgIX5Zn/view?usp=sharing) and [nodes.dmp](https://drive.google.com/file/d/1D4g7PP-mdP9xqsxM9ZC_Bz9wqANkf6UN/view?usp=sharing). These are sequences from the public databases NCBI and BOLD and sequences obtained by scientific groups in University of Barcelona, Center for Advanced Studies of Blanes and Alfred Wagener Institute.

To clean and format the database, the following steps were done: 
- Merge the taxonomic information and cut the region defined by the Leray-XT primers with the class 'SequenceFormatter'.
- Clean the data by removing duplicated, sequences with non-standard bases and filter sequences with too much GC bases with the class 'TaxonomyDataCleaner'.
- finish the process by filtering by length (299-320) and define the classes to predict at different levels by selecting those classes more abundant and merging the others into Others as a prof of concept.

To replicate the creation of the database run the following command from a unix terminal:
```
python ...
```

### Testing the architechtures