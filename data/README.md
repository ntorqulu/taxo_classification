# Database

The database was obtained initially from the [NJORDR](https://github.com/adriantich/NJORDR-MJOLNIR3) repo which attempted to create a database for the COI barcode using the [mkCOInr](https://github.com/meglecz/mkCOInr) repo.
The initial files can be downloaded from [google drive](https://drive.google.com/drive/folders/1NbJ90I6jyYmkxZyJvI70vmsjzPalVbEL?usp=drive_link).

To create the final database the Pipeline4FinalDataset.py script was used.
The classes for each level were defined after inspecting the original data (see overview)

# database columns
- *seqID*: Unique identification for each instance\
- *taxID*: Numeric Id for the scientific_name\
- *scientific_name*: the most precise classification for the instance\
- *sequence*: DNA sequence. This is the input information for the model to process after being coded\
- *superkingdom_name*: superkingdom name. this is the highest taxonomic rank.\
- *kingdom_name*: kingdom name. The different kingdoms belong to different superkingdoms\
- *phylum_name*: phylum name. The different Phyla belong to different kingdoms.\
- *class_name*: class name. The different classes belong to different Phyla.\
- *order_name*:  order name. The different orders belong to different classes.\
- *family_name*: family name. The different families belong to different orders.\
- *genus_name*: genus name. The different genus belong to different families.\
- *species_name*: species name. The different species belong to different genus.\
- *level_1*: First level of labels to predict, the classes belong to the kingdom level and enclosed in the "Eukaryota" Superkingdom --> 'Metazoa', 'Viridiplantae','Fungi','Other_euk','No_euk'\
- *level_2*: Second level of labels to predict, the classes belong to the Phylum level and enclosed in the "Metazoa" Kingdom --> 'Arthropoda','Chordata','Mollusca','Annelida','Echinodermata','Platyhelminthes','Cnidaria','Other_metazoa','No_metazoa'\
- *level_3*: Third level of labels to predict, the classes belong to the Class level and enclosed in the "Arthropoda" phylum --> 'Insecta', 'Arachnida', 'Malacostraca', 'Collembola', 'Hexanauplia','Thecostraca', 'Branchiopoda', 'Diplopoda', 'Ostracoda', 'Chilopoda', 'Pycnogonida','Other_arthropoda','No_arthropoda'\
- *level_4*: Fourth level of labels to predict, the classes belong to the Order level and enclosed in the "Insecta" Class --> 'Diptera', 'Lepidoptera', 'Hymenoptera', 'Coleoptera', 'Hemiptera','Trichoptera', 'Orthoptera', 'Ephemeroptera', 'Odonata', 'Blattodea','Thysanoptera', 'Psocoptera', 'Plecoptera', 'Neuroptera','Other_insecta','No_insecta'\
