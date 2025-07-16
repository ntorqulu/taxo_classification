#!/usr/bin/env Rscript
# install optparse if not installed
if (!requireNamespace("optparse", quietly = TRUE)) {
    install.packages("optparse")
}
library(optparse)

option_list <- list(
    make_option(c("--input_table"), type="character", help="Path to input table"),
    make_option(c("--level"), type="character", help="Taxonomic level"),
    make_option(c("--output_plot"), type="character", help="Path to output plot")
)

opt_parser <- OptionParser(option_list=option_list)
opt <- parse_args(opt_parser)

input_table <- opt$input_table
level_name <- opt$level
output_plot <- opt$output_plot

# input_table <- "PCA/embeddings.tsv"  # Default value for input_table
# level_name <- "genus_name"            # Default value for level_name
# output_plot <- paste0("PCA/",gsub("_name","",level_name),".png")     # Default value for output_plot

df <- read.table(input_table, header = TRUE, sep = "\t", stringsAsFactors = FALSE, check.names = FALSE)
if (ncol(df) == 1) {
    stop("Input file does not appear to be tab-separated. Please provide a valid TSV file.")
}

# Check if the specified level exists in the data frame
if (!(level_name %in% colnames(df))) {
    stop(paste("The specified level", level_name, "does not exist in the input table."))
}

# check that there are at least two columns named with the pattern pca_*_1	and pca_*_2
pca_columns <- grep("^pca_.*_1$", colnames(df), value = TRUE)
if (length(pca_columns) < 1) {
    stop("The input table must contain at least two columns named with the pattern pca_*_1 and pca_*_2.")
}
pca_columns_2 <- grep("^pca_.*_2$", colnames(df), value = TRUE)
if (length(pca_columns_2) < 1) {
    stop("The input table must contain at least two columns named with the pattern pca_*_2.")
}


library(ggplot2)

color_palette <- c(
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)
shape_palette <- c(16, 17, 15, 3, 7, 8)

num_labels <- length(unique(df[[level_name]]))
if (num_labels > 60) {
    stop("Too many labels for unique color/shape combinations (max 60).")
}

# Generate all unique color-shape combinations
combos <- expand.grid(color = color_palette, shape = shape_palette)
combos <- combos[1:num_labels, ]

# Assign each label a unique color-shape pair
labels <- sort(unique(df[[level_name]]))
label_map <- data.frame(
    label = labels,
    color = combos$color,
    shape = combos$shape,
    stringsAsFactors = FALSE
)
df$.__tmp_label__ <- df[[level_name]]
df$level_name <- df[[level_name]]
df <- merge(df, label_map, by.x = ".__tmp_label__", by.y = "label", all.x = TRUE, sort = FALSE)

p <- ggplot(df, aes_string(x = pca_columns[1], y = pca_columns_2[1])) +
    geom_point(aes(color = level_name, shape = level_name), size = 3) +
    scale_color_manual(values = label_map$color) +
    scale_shape_manual(values = label_map$shape) +
    labs(title = paste("PCA Plot at Level:", level_name),
         x = pca_columns[1],
         y = pca_columns_2[1]) +
    theme_minimal() +
    theme(legend.position = "right",
          plot.background = element_rect(color = "white") )
    
print(p)

ggsave(output_plot, p, width = 10, height = 8)
