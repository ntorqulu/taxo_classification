# En aquest script el que faig es retallar les seqs a una regió en concret d'aprox 313
# aquesta regió es selecciona amb uns primers o encebadors que delimiten la regió que volem
# aquests s'anomenen "Leray-XT" i son: 
# fwd --> GGWACWRGWTGRACWNTNTAYCCYCC
# rev --> TANACYTCNGGRTGNCCRAARAAYCA

# el que fa l'script es, a partir de l'arxiu que té totes les sequencies de totes les mides, selecciona aquelles que pot tallar pels primers. Un cop pot definir la regió utilitza aquestes com a centroids per "aliniar" la resta
# i seleccionar-ne la regió en concret.

perl /home/aantich/Nextcloud/2_PROJECTES/NJORDR-MJOLNIR3/mkCOInr/scripts/select_region.pl \
    -tsv /home/aantich/Nextcloud/2_PROJECTES/NJORDR-MJOLNIR3/NJORDR_COI/COMPLETE_DB/NJORDR_sequences.tsv \
    -outdir /home/aantich/Nextcloud/2_PROJECTES/AIDL-project/taxo_classification/data/raw/ \
    -fw 'GGWACWRGWTGRACWNTNTAYCCYCC' \
    -rv 'TANACYTCNGGRTGNCCRAARAAYCA' \
    -e_pcr 1 -min_amplicon_length 299 -max_amplicon_length 320