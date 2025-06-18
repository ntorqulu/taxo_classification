"""Constants used throughout the project."""

TAXONOMY_LEVELS: list[str] = [
    'kingdom_name',
    'phylum_name',
    'class_name',
    'order_name'
]

# Taxonomy classification labels for different taxonomic levels
TAXONOMY_LABELS: dict[str, list[str]]= {
    "kingdom_name": [
        'Metazoa', 'Viridiplantae', 'Fungi', 'Other_euk', 'No_euk'
    ],
    "phylum_name": [
        'Arthropoda', 'Chordata', 'Mollusca', 'Annelida', 'Echinodermata', 
        'Platyhelminthes', 'Cnidaria', 'Other_metazoa', 'No_metazoa'
    ],
    "class_name": ['Insecta', 'Arachnida', 'Malacostraca', 'Collembola', 'Hexanauplia', 
           'Thecostraca', 'Branchiopoda', 'Diplopoda', 'Ostracoda', 'Chilopoda', 'Pycnogonida',
           'Other_arthropoda','No_arthropoda'
    ],
    "order_name": ['Diptera', 'Lepidoptera', 'Hymenoptera', 'Coleoptera', 'Hemiptera', 
           'Trichoptera', 'Orthoptera', 'Ephemeroptera', 'Odonata', 'Blattodea', 
           'Thysanoptera', 'Psocoptera', 'Plecoptera', 'Neuroptera',
           'Other_insecta','No_insecta'
    ],
}

def get_class_names(label_column_name: str) -> list:
    """
    Get the class names for a specific taxonomic level.
    
    Args:
        label_column_name: The name of the taxonomic level column (e.g., 'phylum_name')
        
    Returns:
        List of class names for that taxonomic level
    """
    if label_column_name in TAXONOMY_LABELS:
        return TAXONOMY_LABELS[label_column_name]
    else:
        # Return empty list or default if not found
        return []