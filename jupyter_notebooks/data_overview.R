#
# faig una analisi ràpida de les dades que tenim

data <- read.csv('data/merged/final_taxonomy.csv')

data$sequence <- toupper(data$sequence)

library(stringr)

data <- data[str_length(data$sequence) %in% 290:320,] 

data <- data[!grepl('N|Y|R|W|K|S|M|D', data$sequence),]

#
# quants forats tenim (rang taxonomic buit)
#

# sequencies totals
nrow(data)
# [1] 2990800

# sequencies que no arriben a espècie
sum(data$species_name == "")
# [1] 1547501
sum(data$species_name == "") / nrow(data) *100
# [1] 51.74204 percent

# que no arriben a genere
sum(data$genus_name == "" & data$species_name == "")  / nrow(data) *100
# [1] 33.61786 percent

# que no a familia
sum(data$family_name == "" & data$genus_name == "" & data$species_name == "")  / nrow(data) *100
# [1] 4.86843

# que no a ordre
sum(data$order_name == "" & data$family_name == "" & data$genus_name == "" & data$species_name == "")  / nrow(data) *100
# [1] 0.3602715

# que no a classe
sum(data$class_name == "" & data$order_name == "" & data$family_name == "" & data$genus_name == "" & data$species_name == "")  / nrow(data) *100
# [1] 0.04125986

# que no a classe
sum(data$phylum_name == "" & data$class_name == "" & data$order_name == "" & data$family_name == "" & data$genus_name == "" & data$species_name == "")  / nrow(data) *100
# [1] 0.01243814


library(ggsankey)

df <- data %>%
  make_long(superkingdom_name, kingdom_name,phylum_name, class_name)

library(ggplot2)
library(dplyr) # Also needed

p <- ggplot(df, aes(x = x, 
               next_x = next_x, 
               node = node, 
               next_node = next_node))+
            #    next_node = next_node,
            #    fill = factor(node),
            #    label = node)) +
  geom_sankey(flow.alpha = 0.75, node.color = 1) +
#   geom_sankey_label() +
  theme_sankey(base_size = 16)
ggsave('plot.png', plot = p)

# > as.data.frame(sort(table(data$phylum_name[data$kingdom_name == 'Metazoa']), decreasing=T))
#               Var1    Freq
# 1       Arthropoda 2432311
# 2         Chordata  281063
# 3         Mollusca  132226
# 4         Annelida   39733
# 5    Echinodermata   19753
# 6  Platyhelminthes   12942
# 7         Cnidaria   10917
# 8         Nematoda    9728
# 9         Rotifera    5855
# 10        Porifera    3293
# 11        Nemertea    2713
# 12      Tardigrada    1610
# 13     Onychophora    1282
# 14         Bryozoa    1152
# 15  Acanthocephala    1051
# 16       Sipuncula     639
# 17    Chaetognatha     534
# 18      Ctenophora     385
# 19     Kinorhyncha     212
# 20     Brachiopoda     194
# 21 Xenacoelomorpha     194
# 22    Nematomorpha     181
# 23    Gastrotricha     177
# 24    Hemichordata     172
# 25     Cycliophora     109
# 26       Phoronida      70
# 27      Priapulida      52
# 28      Entoprocta      21
# 29 Gnathostomulida      14
# 30        Placozoa      10
# 31                       7
# 32    Orthonectida       1

# >  as.data.frame(sort(table(data$class_name[data$phylum_name == 'Arthropoda']), decreasing=T))
#             Var1    Freq
# 1        Insecta 2110562
# 2      Arachnida  149944
# 3   Malacostraca   91149
# 4     Collembola   34004
# 5    Hexanauplia   14108
# 6    Thecostraca   12523
# 7   Branchiopoda    8512
# 8      Diplopoda    3457
# 9      Chilopoda    2657
# 10     Ostracoda    2628
# 11   Pycnogonida    1863
# 12                   348
# 13 Ichthyostraca     155
# 14       Protura     142
# 15      Symphyla      74
# 16     Pauropoda      70
# 17     Remipedia      58
# 18   Merostomata      48
# 19 Cephalocarida       8
# 20 Mystacocarida       1

# > as.data.frame(sort(table(data$order_name[data$class_name == 'Insecta']), decreasing=T))
#                          Var1   Freq
# 1                     Diptera 850606
# 2                 Lepidoptera 420961
# 3                 Hymenoptera 351900
# 4                  Coleoptera 244078
# 5                   Hemiptera 104436
# 6                 Trichoptera  28033
# 7                  Orthoptera  23278
# 8               Ephemeroptera  16489
# 9                     Odonata  11114
# 10                  Blattodea  10050
# 11               Thysanoptera   9572
# 12                 Psocoptera   9475
# 13                 Plecoptera   8873
# 14                 Neuroptera   8096
# 15 Psocodea (from superorder)   2661
# 16                Phasmatodea   1772
# 17                   Mantodea   1768
# 18              Archaeognatha   1242
# 19                Megaloptera   1221
# 20               Phthiraptera   1049
# 21               Siphonaptera    947
# 22                  Mecoptera    939
# 23                 Dermaptera    713
# 24               Strepsiptera    456
# 25                 Embioptera    312
# 26                  Zygentoma    193
# 27                               140
# 28              Raphidioptera    119
# 29           Mantophasmatodea     38
# 30                  Zoraptera     25
# 31            Grylloblattodea      6

# filtro la taula
# elimino seqs que no arriben a order_name
data <- data[!(data$order_name == "" & data$family_name == "" & data$genus_name == "" & data$species_name == ""),]

# nivell 1 eukariotes
euk_class <- c('Metazoa', 'Viridiplantae','Fungi','No_euk', 'Others')

data$euk_class <- data$kingdom_name
data$euk_class[grepl('Bacteria',data$kingdom_name)] <- 'No_euk'
data$euk_class[!(data$euk_class %in% euk_class)] <- 'Others'

# nivell 2 Metazoa
metazoa_class <- c('Arthropoda','Chordata','Mollusca','Annelida','Echinodermata','Platyhelminthes','Cnidaria',
    'No_metazoa', 'Others')

data$metazoa_class <- data$phylum_name
data$metazoa_class[!grepl('Metazoa',data$euk_class)] <- 'No_metazoa'
data$metazoa_class[!(data$metazoa_class %in% metazoa_class)] <- 'Others'

# nivell 3 Arthropoda
arthropoda_class <- c('Insecta', 'Arachnida', 'Malacostraca', 'Collembola', 'Hexanauplia', 
    'Thecostraca', 'Branchiopoda', 'Diplopoda', 'Ostracoda', 'Chilopoda', 'Pycnogonida',
    'No_arthropoda', 'Others')

data$arthropoda_class <- data$class_name
data$arthropoda_class[!grepl('Arthropoda',data$metazoa_class)] <- 'No_arthropoda'
data$arthropoda_class[!(data$arthropoda_class %in% arthropoda_class)] <- 'Others'

# nivell 4 Insecta
insecta_class <- c('Diptera', 'Lepidoptera', 'Hymenoptera', 'Coleoptera', 'Hemiptera', 
    'Trichoptera', 'Orthoptera', 'Ephemeroptera Odonata', 'Blattodea', 
    'Thysanoptera', 'Psocoptera', 'Plecoptera', 'Neuroptera',
    'No_insecta', 'Others')

data$insecta_class <- data$order_name
data$insecta_class[!grepl('Insecta',data$arthropoda_class)] <- 'No_insecta'
data$insecta_class[!(data$insecta_class %in% insecta_class)] <- 'Others'

write.csv(data, 'data/merged/new_classes.csv', row.names = F)

df2 <- data %>%
  make_long(euk_class, metazoa_class, arthropoda_class, insecta_class)

lev <- unique(c(euk_class, metazoa_class, arthropoda_class, insecta_class))

df2$node2 <- factor(df2$node, levels = lev, labels = lev)


p <- ggplot(df2, aes(x = x, 
               next_x = next_x, 
               node = node2, 
            #    next_node = next_node))+
               next_node = next_node,
               fill = node2,
               label = node)) +
  geom_sankey(flow.alpha = 0.75, node.color = 1) +
  geom_sankey_label() +
  theme_sankey(base_size = 16)
ggsave('plot2.png', plot = p)

pp <- ggplot(df2, aes(x = x, 
               next_x = next_x, 
               node = node2, 
            #    next_node = next_node))+
               next_node = next_node,
               fill = node2,
               label = node)) +
  geom_alluvial(flow.alpha = .6) +
  geom_alluvial_text(size = 3, color = "black") +
  scale_fill_viridis_d(drop = FALSE) +
  theme_alluvial(base_size = 18) +
  labs(x = NULL) +
  theme(legend.position = "none",
        plot.title = element_text(hjust = .5))
ggsave('plot3.png', plot = pp)

