from dataset.utils import info, warn

"""Constants used throughout the project."""


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


def get_class_id(level_name: str, value: str) -> int:
    if level_name not in TAXONOMY_LABELS:
        raise ValueError(f"Invalid taxonomic level: {level_name}")
    if value not in TAXONOMY_LABELS[level_name]:
        raise ValueError(f"Invalid value for {level_name}: {value}")
    class_id = TAXONOMY_LABELS[level_name].index(value)
    return class_id


def get_class_name(level_name: str, class_id: int) -> str:
    if level_name not in TAXONOMY_LABELS:
        raise ValueError(f"Invalid taxonomic level: {level_name}")
    if class_id < 0 or class_id >= len(TAXONOMY_LABELS[level_name]):
        raise ValueError(f"Invalid class_id: {class_id}")
    class_name = TAXONOMY_LABELS[level_name][class_id]
    return class_name

def get_max_class_id(level_name: str) -> int:
    if level_name not in TAXONOMY_LABELS:
        raise ValueError(f"Invalid taxonomic level: {level_name}")
    return len(TAXONOMY_LABELS[level_name]) - 1


TAXONOMY_LEVELS: list[str] = list(TAXONOMY_LABELS.keys())

def wrong_class_values(level_name: str, values: list | dict) -> None | dict[str, list[str]]:
    if level_name not in TAXONOMY_LABELS:
        raise ValueError(f"Invalid taxonomic level: {level_name}")

    if isinstance(values, dict):
        values = list(values.keys())
    values = set(values)

    level_values = set(TAXONOMY_LABELS[level_name])

    missing_values = level_values - values
    unknown_values = values - level_values

    if not missing_values and not unknown_values:
        return None

    return {
        'missing': list(missing_values),
        'unknown': list(unknown_values)
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