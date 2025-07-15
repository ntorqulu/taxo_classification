# Load necessary libraries
library(ggplot2)
library(readr)

# Import the data
data <- read_csv("/home/aantich/Nextcloud/2_PROJECTES/AIDL-project/taxo_classification/Results/PLOTS/Results_arch_coding.csv")

# Create the first bar plot
plot1 <- ggplot(data[data$Plot == "A",], aes(x = Architecture, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white"),
          axis.text.x = element_text(colour = "black", size = 12, angle = 45, hjust = 0.8),
          legend.text = element_text(size = 14)) +
    labs(title = "Genus",
         y = "")

print(plot1)

ggsave("/home/aantich/Nextcloud/2_PROJECTES/AIDL-project/taxo_classification/Results/PLOTS/arch_genus.png", plot = plot1)

# Create the second bar plot
plot2 <- ggplot(data[data$Plot == "B",], aes(x = coding, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white"),
          axis.text.x = element_text(colour = "black", size = 12, angle = 45, hjust = 0.8),
          legend.text = element_text(size = 14)) +
    labs(title = "Genus",
         y = "")
print(plot2)
ggsave("/home/aantich/Nextcloud/2_PROJECTES/AIDL-project/taxo_classification/Results/PLOTS/coding_genus.png", plot = plot2)

# Create the third bar plot
plot3 <- ggplot(data[data$Plot == "C",], aes(x = coding, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white"),
          axis.text.x = element_text(colour = "black", size = 12, angle = 45, hjust = 0.8),
          legend.text = element_text(size = 14)) +
    labs(title = "Order",
         y = "")
print(plot3)
ggsave("/home/aantich/Nextcloud/2_PROJECTES/AIDL-project/taxo_classification/Results/PLOTS/coding_order.png", plot = plot3)

# Create the fourth bar plot
plot4 <- ggplot(data[data$Plot == "D",], aes(x = Architecture, y = Value, fill = Metric)) +
    geom_bar(stat = "identity", position = "dodge") +
    theme_minimal() +
    theme(plot.background = element_rect(fill = "white"),
          axis.text.x = element_text(colour = "black", size = 12, angle = 45, hjust = 0.8),
          legend.text = element_text(size = 14)) +
    labs(title = "Order",
         y = "")
print(plot4)
ggsave("/home/aantich/Nextcloud/2_PROJECTES/AIDL-project/taxo_classification/Results/PLOTS/arch_order.png", plot = plot4)
