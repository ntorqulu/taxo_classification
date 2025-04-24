import pandas as pd

# File trimmed.tsv has columns: seqID, taxID, sequence
pd_trimmed = pd.read_csv('data/raw/trimmed.tsv', sep='\t', header=0, names=['seqID', 'taxID', 'sequence'], dtype=str)
pd_trimmed['seqID'] = pd_trimmed['seqID'].astype(str)
pd_trimmed['taxID'] = pd_trimmed['taxID'].astype(str)
pd_trimmed['sequence'] = pd_trimmed['sequence'].astype(str).str.upper()
print(pd_trimmed.head())
pd_trimmed.to_csv('data/raw/trimmed.csv', index=False)

# Read names.dmp file and store it in a pandas dataframe
pd_names = pd.read_csv('data/raw/names.dmp', sep='\t\|\t', 
                      engine='python',
                      header=None, 
                      names=['tax_id', 'name_txt', 'unique_name', 'name_class'])
pd_names['name_class'] = pd_names['name_class'].str.replace('\t\|$', '', regex=True)
print("\nTaxonomy Names Data:")
print(pd_names.head())
pd_names.to_csv('data/raw/names.csv', index=False)

# Read nodes.dmp file and store it in a pandas dataframe
pd_nodes = pd.read_csv('data/raw/nodes.dmp', sep='\t\|\t', 
                      engine='python',
                      header=None, 
                      names=['tax_id', 'parent_tax_id', 'rank', 'embl_code', 'division_id', 
                             'inherited_div_flag', 'genetic_code_id', 'inherited_GC_flag',
                             'mitochondrial_genetic_code_id', 'inherited_MGC_flag',
                             'GenBank_hidden_flag', 'hidden_subtree_root_flag', 'comments'])

# Clean up the trailing "|" with tab
pd_nodes['comments'] = pd_nodes['comments'].str.replace('\t\|$', '', regex=True)
print("\nTaxonomy Nodes Data:")
print(pd_nodes.head())
pd_nodes.to_csv('data/raw/nodes.csv', index=False)

sci_names = pd_names[pd_names['name_class'] == 'scientific name']
sci_names_dict = dict(zip(sci_names['tax_id'].astype(str), sci_names['name_txt']))

parent_dict = dict(zip(pd_nodes['tax_id'].astype(str), pd_nodes['parent_tax_id'].astype(str)))
rank_dict = dict(zip(pd_nodes['tax_id'].astype(str), pd_nodes['rank']))

taxonomy_ranks = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']

def get_lineage(taxid):
    lineage = {rank: None for rank in taxonomy_ranks}
    current_id = str(taxid)
    visited = set()
    while current_id and current_id not in visited:
        visited.add(current_id)
        # Get rank and name for current node
        rank = rank_dict.get(current_id)
        name = sci_names_dict.get(current_id)
        
        if rank in lineage and name:
            lineage[rank] = name
        
        # Move to parent node
        parent_id = parent_dict.get(current_id)
        if not parent_id or parent_id == current_id:
            break
        current_id = parent_id
    
    return lineage

merged_df = pd_trimmed.copy()

merged_df['scientific_name'] = merged_df['taxID'].map(sci_names_dict)
merged_df['rank'] = merged_df['taxID'].map(rank_dict)

for rank in taxonomy_ranks:
    merged_df[f'{rank}_name'] = None

print("\nProcessing taxonomic lineage for each sequence...")
lineage_cache = {}
for taxid in merged_df['taxID'].unique():
    lineage_cache[taxid] = get_lineage(taxid)

for idx, row in merged_df.iterrows():
    taxid = row['taxID']
    lineage = lineage_cache.get(taxid, {})
    
    for rank, name in lineage.items():
        merged_df.at[idx, f'{rank}_name'] = name

final_columns = ['seqID', 'taxID', 'scientific_name', 'sequence'] + [f'{rank}_name' for rank in taxonomy_ranks]

final_df = merged_df[final_columns]

print("\nFinal DataFrame:")
final_df.to_csv('data/merged/final_taxonomy.csv', index=False)
print(final_df.head())