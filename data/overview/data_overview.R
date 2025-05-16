#
# faig una analisi ràpida de les dades que tenim

data <- read.csv('data/processed/cleaned_sequences.csv')

data$sequence <- toupper(data$sequence)

library(stringr)

data <- data[str_length(data$sequence) %in% 290:320,] 

data <- data[!grepl('N|Y|R|W|K|S|M|D', data$sequence),]

# > as.data.frame(sort(table(data$kingdom_name),decreasing = T))
#            Var1   Freq
# 1       Metazoa 603146
# 2 Viridiplantae    517
# 3         Fungi    325

# > as.data.frame(sort(table(data$phylum_name[data$kingdom_name == 'Metazoa']), decreasing=T))
#               Var1   Freq
# 1       Arthropoda 444318
# 2         Chordata  94170
# 3         Mollusca  36378
# 4         Annelida   7512
# 5    Echinodermata   6394
# 6         Nematoda   4016
# 7  Platyhelminthes   2814
# 8         Cnidaria   2636
# 9         Rotifera   1713
# 10        Porifera    717
# 11     Onychophora    601
# 12        Nemertea    403
# 13  Acanthocephala    385
# 14      Tardigrada    305
# 15         Bryozoa    244
# 16       Sipuncula    234
# 17    Chaetognatha    109
# 18      Ctenophora     81
# 19     Brachiopoda     77
# 20    Nematomorpha     31
# 21      Priapulida      6
# 22    Hemichordata      2

# >  as.data.frame(sort(table(data$class_name[data$phylum_name == 'Arthropoda']), decreasing=T))
#             Var1   Freq
# 1        Insecta 375729
# 2      Arachnida  29225
# 3   Malacostraca  24371
# 4    Hexanauplia   3636
# 5    Thecostraca   3536
# 6     Collembola   3507
# 7   Branchiopoda   2047
# 8      Chilopoda    722
# 9      Diplopoda    608
# 10     Ostracoda    580
# 11   Pycnogonida    203
# 12       Protura     64
# 13 Ichthyostraca     53
# 14   Merostomata     17
# 15     Remipedia     16
# 16 Cephalocarida      3
# 17     Pauropoda      1

# > as.data.frame(sort(table(data$order_name[data$class_name == 'Insecta']), decreasing=T))
#                Var1   Freq
# 1       Lepidoptera 128347
# 2           Diptera  76824
# 3        Coleoptera  66891
# 4       Hymenoptera  44235
# 5         Hemiptera  21136
# 6       Trichoptera  10195
# 7        Orthoptera   7069
# 8           Odonata   4687
# 9     Ephemeroptera   4560
# 10       Plecoptera   3163
# 11       Neuroptera   1778
# 12        Blattodea   1664
# 13     Thysanoptera   1235
# 14      Phasmatodea    812
# 15       Psocoptera    709
# 16      Megaloptera    477
# 17        Mecoptera    475
# 18         Mantodea    431
# 19     Phthiraptera    377
# 20     Siphonaptera    285
# 21       Dermaptera     96
# 22    Archaeognatha     71
# 23       Embioptera     61
# 24        Zygentoma     40
# 25     Strepsiptera     38
# 26    Raphidioptera     31
# 27 Mantophasmatodea     29
# 28        Zoraptera      9
# 29  Grylloblattodea      4

# filtro la taula
# elimino seqs que no arriben a order_name
data <- data[!(data$order_name == "" & data$family_name == "" & data$genus_name == "" & data$species_name == ""),]

# nivell 1 eukariotes
level_1 <- c('Metazoa', 'Viridiplantae','Fungi',
    'Other_euk','No_euk')

data$level_1 <- data$kingdom_name
data$level_1[grepl('Bacteria',data$kingdom_name)] <- 'No_euk'
data$level_1[!(data$level_1 %in% level_1)] <- 'Other_euk'

# nivell 2 Metazoa
level_2 <- c('Arthropoda','Chordata','Mollusca','Annelida','Echinodermata','Platyhelminthes','Cnidaria',
    'Other_metazoa','No_metazoa')

data$level_2 <- data$phylum_name
data$level_2[!grepl('Metazoa',data$level_1)] <- 'No_metazoa'
data$level_2[!(data$level_2 %in% level_2)] <- 'Other_metazoa'

# nivell 3 Arthropoda
level_3 <- c('Insecta', 'Arachnida', 'Malacostraca', 'Collembola', 'Hexanauplia', 
    'Thecostraca', 'Branchiopoda', 'Diplopoda', 'Ostracoda', 'Chilopoda', 'Pycnogonida',
    'Other_arthropoda','No_arthropoda')

data$level_3 <- data$class_name
data$level_3[!grepl('Arthropoda',data$level_2)] <- 'No_arthropoda'
data$level_3[!(data$level_3 %in% level_3)] <- 'Other_arthropoda'

# nivell 4 Insecta
level_4 <- c('Diptera', 'Lepidoptera', 'Hymenoptera', 'Coleoptera', 'Hemiptera', 
    'Trichoptera', 'Orthoptera', 'Ephemeroptera', 'Odonata', 'Blattodea', 
    'Thysanoptera', 'Psocoptera', 'Plecoptera', 'Neuroptera',
    'Other_insecta','No_insecta')

data$level_4 <- data$order_name
data$level_4[!grepl('Insecta',data$level_3)] <- 'No_insecta'
data$level_4[!(data$level_4 %in% level_4)] <- 'Other_insecta'

# write.csv(data, 'data/database.csv', row.names = F)
# data <- read.csv('data/database.csv')

data$level_1 <- factor(data$level_1, levels = level_1, labels = level_1)
data$level_2 <- factor(data$level_2, levels = level_2, labels = level_2)
data$level_3 <- factor(data$level_3, levels = level_3, labels = level_3)
data$level_4 <- factor(data$level_4, levels = level_4, labels = level_4)

table(data$level_1)
data_plot <- rbind(
  cbind(level = 'level_1',
        as.data.frame(table(data$level_1))),
  cbind(level = 'level_2',
        as.data.frame(table(data$level_2))),
  cbind(level = 'level_3',
        as.data.frame(table(data$level_3))),
  cbind(level = 'level_4',
        as.data.frame(table(data$level_4)))
)
data_plot$percentage <- data_plot$Freq / nrow(data) * 100

library(ggplot2)

p <- ggplot(data_plot, aes(x = level, 
               y = percentage, 
               fill = Var1)) +
  geom_bar(stat = 'identity', color = 'black') +
  scale_fill_manual(
    values = c(
      'Metazoa'='blue',
      'Arthropoda'='red',
      'Insecta'='green',
      'Other_euk'='white',
      'No_euk'='black',
      'Other_metazoa'='white',
      'No_metazoa'='black',
      'Other_arthropoda'='white',
      'No_arthropoda'='black',
      'Other_insecta'='white',
      'No_insecta'='black')
    ) +
  coord_polar(theta = 'y') 

ggsave('data/overview/pie_plot.png', plot = p)


library(ggsankey)
df2 <- data %>%
  make_long(level_1, level_2, level_3, level_4)

lev <- unique(c(level_1, level_2, level_3, level_4))

df2$node2 <- factor(df2$node, levels = lev, labels = lev)
df2$next_node2 <- factor(df2$next_node, levels = lev, labels = lev)
# sort as following the levels
df2 <- df2[order(df2$next_node2),] 


p <- ggplot(df2, aes(x = x, 
               next_x = next_x, 
               node = node2, 
            #    next_node = next_node))+
               next_node = next_node2,
               fill = node2,
               label = node)) +
  geom_sankey(flow.alpha = 0.75, node.color = 1) +
  geom_sankey_label() +
  theme_sankey(base_size = 16)+
  theme(legend.position = "none")
ggsave('data/overview/connection_plot.png', plot = p)

pp <- ggplot(df2, aes(x = x, 
               next_x = next_x, 
               node = node2, 
            #    next_node = next_node))+
               next_node = next_node2,
               fill = node2,
               label = node)) +
  geom_alluvial(flow.alpha = .6) +
  geom_alluvial_text(size = 3, color = "black") +
  scale_fill_viridis_d(drop = FALSE) +
  theme_alluvial(base_size = 18) +
  labs(x = NULL) +
  theme(legend.position = "none",
        plot.title = element_text(hjust = .5))
ggsave('data/overview/plot3.png', plot = pp)

