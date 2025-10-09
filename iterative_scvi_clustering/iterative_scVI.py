import scanpy as sc
import scvi
import leidenalg
import numpy as np
import igraph
import pandas as pd
import logging
def Iterative_Clustering_scVI(adata, ndims=30, num_iterations=20, min_pct=0.4, min_log2_fc=2, batch_size=2048, min_bayes_score=8, min_cluster_size=4, model=None, embedding_key='X_scVI'):
    """
    Wrapper function to perform iterative clustering using scVI and Leiden algorithm.
    Args:
        adata: AnnData object containing the scRNA-seq data with the specified embedding in obsm.
        ndims: Number of latent dimensions to use from the embedding.
        num_iterations: Maximum number of clustering iterations.
        min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
        min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
        batch_size: Batch size for scVI differential expression.
        min_bayes_score: Minimum score for a gene to be considered differentially expressed.
        min_cluster_size: Minimum size of clusters to retain.
        model: scVI model object for differential expression analysis.
        embedding_key: Key in adata.obsm indicating the embedding to use (default: 'X_scVI').
    Returns:
        adata: AnnData object with updated clustering in adata.obs['leiden'].
    """
    adata.obs['leiden']='1'
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    previous_num_clusters = 1
    for i in range(num_iterations):
        adata = Clustering_Iteration(adata, ndims=ndims, min_pct=min_pct, min_log2_fc=min_log2_fc, batch_size=batch_size, min_bayes_score=min_bayes_score, model=model, embedding_key=embedding_key)
        if len(adata.obs['leiden'].cat.categories) == previous_num_clusters:
            break
        previous_num_clusters = len(adata.obs['leiden'].cat.categories)
    return adata

def Find_Nearest_Cluster(centroids, cluster_labels, target_cluster):
    """
    Find the nearest cluster to the target cluster based on centroid distance.
    Args:
        centroids: Precomputed centroids array (n_clusters x n_dims)
        cluster_labels: List of cluster labels corresponding to centroid rows
        target_cluster: The cluster to find the nearest neighbor for
    Returns:
        nearest_cluster: The label of the nearest cluster, or None if no suitable cluster found
    """
    from sklearn.metrics import pairwise_distances
    
    # Get all clusters except the target cluster
    other_clusters = [c for c in cluster_labels if c != target_cluster]
    
    if len(other_clusters) == 0:
        return None
    
    try:
        # Find the index of target cluster and other clusters
        cluster_to_idx = {cluster: i for i, cluster in enumerate(cluster_labels)}
        
        if target_cluster not in cluster_to_idx:
            return None
            
        target_idx = cluster_to_idx[target_cluster]
        
        # Calculate distances from target cluster to all other clusters
        target_centroid = centroids[target_idx:target_idx+1]  # Keep as 2D array
        other_centroids = np.array([centroids[cluster_to_idx[c]] for c in other_clusters if c in cluster_to_idx])
        
        if len(other_centroids) == 0:
            return None
            
        # Calculate distances
        distances = pairwise_distances(target_centroid, other_centroids)[0]
        
        # Find nearest cluster
        nearest_idx = np.argmin(distances)
        nearest_cluster = other_clusters[nearest_idx]
        
        return nearest_cluster
        
    except Exception as e:
        print(f"Error finding nearest cluster for {target_cluster}: {e}")
        # Fallback: return the first available cluster
        return other_clusters[0] if other_clusters else None

def Clustering_Iteration(adata, ndims=30, min_pct=0.4, min_log2_fc=2, batch_size=2048, min_bayes_score=8, min_cluster_size=4, model=None, embedding_key='X_scVI'):
    """
    Performs one iteration of clustering and merging.
    Args:
         adata: AnnData object containing the scRNA-seq data with the specified embedding in obsm.
         ndims: Number of latent dimensions to use from the embedding.
         min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
         min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
         batch_size: Batch size for scVI differential expression.
         min_bayes_score: Minimum score for a gene to be considered differentially expressed.
         min_cluster_size: Minimum size of clusters to retain.
         model: scVI model object for differential expression analysis. If None, clustering will still occur but differential expression scoring will be skipped.
         embedding_key: Key in adata.obsm indicating the embedding to use (default: 'X_scVI').
    Returns:
         adata: AnnData object with updated clustering in adata.obs['leiden'].
    """
    
    clusters = adata.obs['leiden'].cat.categories.copy()
    
    for cluster in clusters:
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_adata = adata[cluster_mask].copy()
        
        if cluster_adata.n_obs < min_cluster_size:
            continue
            
        if cluster_adata.n_obs < 15:
            sc.pp.neighbors(cluster_adata, use_rep=embedding_key, n_neighbors=int(np.floor(cluster_adata.n_obs/2)), n_pcs=ndims)
        else:
            sc.pp.neighbors(cluster_adata, use_rep=embedding_key, n_pcs=ndims)
        g = sc._utils.get_igraph_from_adjacency(cluster_adata.obsp['connectivities'])
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
        cluster_adata.obs['leiden'] = [str(c) for c in part.membership]
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].astype('category')
        
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
        
        sub_clusters = cluster_adata.obs['leiden'].cat.categories
        nonempty_sub_clusters = [subcluster for subcluster in sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
        
        if len(nonempty_sub_clusters) < 2:
            continue
            
        changes_made = True
        merged_pairs = []
        
        while changes_made:
            changes_made = False
            
            cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
            
            sub_clusters = cluster_adata.obs['leiden'].cat.categories
            nonempty_sub_clusters = [subcluster for subcluster in sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
            
            if len(nonempty_sub_clusters) < 2:
                break
            centroids = Find_Centroids(cluster_adata, cluster_key='leiden', embedding_key=embedding_key, ndims=ndims)
            
            if centroids.shape[0] < 2:
                break
                
            centroid_map = {subcluster: i for i, subcluster in enumerate(nonempty_sub_clusters)}
            
            min_distance = np.inf
            closest_pair = None
            
            for sub_cluster in nonempty_sub_clusters:
                if sub_cluster not in centroid_map:
                    continue
                
                # Use the refactored Find_Nearest_Cluster function
                closest_sub_cluster = Find_Nearest_Cluster(centroids, nonempty_sub_clusters, sub_cluster)
                
                if closest_sub_cluster is None:
                    continue
                
                if (sub_cluster, str(closest_sub_cluster)) in merged_pairs or (str(closest_sub_cluster), sub_cluster) in merged_pairs:
                    continue
                
                # Calculate distance between the pair for comparison
                idx = centroid_map[sub_cluster]
                closest_idx = centroid_map.get(closest_sub_cluster)
                
                if closest_idx is None or idx >= centroids.shape[0] or closest_idx >= centroids.shape[0]:
                    continue
                
                from sklearn.metrics import pairwise_distances
                distance = pairwise_distances(centroids[idx:idx+1], centroids[closest_idx:closest_idx+1])[0][0]
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (sub_cluster, closest_sub_cluster)
            
            if closest_pair is None:
                break
                
            sub_cluster, closest_sub_cluster = closest_pair
                
            n_cells_sub = np.sum(cluster_adata.obs['leiden'] == sub_cluster)
            n_cells_closest = np.sum(cluster_adata.obs['leiden'] == closest_sub_cluster)
            
            # Force merge if either cluster is too small (regardless of DE score)
            if n_cells_sub < min_cluster_size or n_cells_closest < min_cluster_size:
                print(f"Force merging small sub-clusters: {sub_cluster} ({n_cells_sub} cells) with {closest_sub_cluster} ({n_cells_closest} cells)")
                cluster_adata.obs.loc[cluster_adata.obs['leiden'] == closest_sub_cluster, 'leiden'] = sub_cluster
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                changes_made = True
                continue
            
            # Skip DE analysis if clusters are too small for reliable DE (but above min_cluster_size)
            if n_cells_sub < 3 or n_cells_closest < 3:
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                continue
                
            # Perform differential expression analysis for larger clusters
            bayes_de_score = Bayes_DE_Score(cluster_adata, sub_cluster, closest_sub_cluster, min_pct, min_log2_fc, batch_size, model=model)
            
            if bayes_de_score < min_bayes_score:
                cluster_adata.obs.loc[cluster_adata.obs['leiden'] == closest_sub_cluster, 'leiden'] = sub_cluster
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                changes_made = True
        
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
        
        # Store cluster mapping for later renaming
        final_sub_clusters = cluster_adata.obs['leiden'].cat.categories
        final_nonempty_sub_clusters = [subcluster for subcluster in final_sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
        
        if len(final_nonempty_sub_clusters) > 1:
            # Store the mapping from old subclusters to original cluster indices for renaming later
            for subcluster in final_nonempty_sub_clusters:
                subcluster_mask = cluster_adata.obs['leiden'] == subcluster
                original_indices = cluster_adata.obs.index[subcluster_mask]
                # Temporarily store with cluster prefix to avoid conflicts
                temp_label = f"temp_{cluster}_{subcluster}"
                adata.obs.loc[original_indices, 'leiden'] = temp_label
    
    adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    
    # Final cleanup: merge any remaining clusters smaller than min_cluster_size
    final_cleanup_changes = True
    while final_cleanup_changes:
        final_cleanup_changes = False
        current_clusters = adata.obs['leiden'].cat.categories.copy()
        
        for cluster in current_clusters:
            cluster_size = np.sum(adata.obs['leiden'] == cluster)
            if cluster_size < min_cluster_size:
                # Find nearest cluster and merge
                other_clusters = [c for c in current_clusters if c != cluster and np.sum(adata.obs['leiden'] == c) > 0]
                if other_clusters:
                    # Calculate centroids for final cleanup
                    cleanup_centroids = Find_Centroids(adata, cluster_key='leiden', embedding_key=embedding_key, ndims=ndims)
                    nearest_cluster = Find_Nearest_Cluster(cleanup_centroids, current_clusters, cluster)
                    if nearest_cluster is not None:
                        print(f"Final cleanup: merging small cluster {cluster} ({cluster_size} cells) with nearest cluster {nearest_cluster}")
                        adata.obs.loc[adata.obs['leiden'] == cluster, 'leiden'] = nearest_cluster
                        final_cleanup_changes = True
                        break  # Start over to avoid modifying categories while iterating
        
        if final_cleanup_changes:
            adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    
    # Final renaming: convert temp labels to proper cluster names
    adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    current_clusters = adata.obs['leiden'].cat.categories.copy()
    
    # Group temp clusters by their parent cluster
    cluster_groups = {}
    for cluster_label in current_clusters:
        if cluster_label.startswith('temp_'):
            # Parse temp_parentcluster_subcluster
            parts = cluster_label.split('_', 2)
            if len(parts) >= 2:
                parent_cluster = parts[1]
                if parent_cluster not in cluster_groups:
                    cluster_groups[parent_cluster] = []
                cluster_groups[parent_cluster].append(cluster_label)
        else:
            # Regular cluster (not temp), treat as single cluster group
            cluster_groups[cluster_label] = [cluster_label]
    
    # Rename each group
    for parent_cluster, temp_labels in cluster_groups.items():
        if len(temp_labels) == 1 and not temp_labels[0].startswith('temp_'):
            # Single regular cluster, no renaming needed
            continue
        elif len(temp_labels) == 1:
            # Single temp cluster, rename to parent cluster name
            adata.obs.loc[adata.obs['leiden'] == temp_labels[0], 'leiden'] = parent_cluster
        else:
            # Multiple subclusters, rename with numbered suffixes
            for i, temp_label in enumerate(temp_labels, 1):
                new_label = f"{parent_cluster}_{i}"
                # Ensure new label is available in categories
                if new_label not in adata.obs['leiden'].cat.categories:
                    adata.obs['leiden'] = adata.obs['leiden'].cat.add_categories([new_label])
                adata.obs.loc[adata.obs['leiden'] == temp_label, 'leiden'] = new_label
    
    adata.obs['leiden'] = adata.obs['leiden'].cat.remove_unused_categories()
    return adata
def Find_Centroids(adata, cluster_key='leiden', embedding_key='X_scVI', ndims=30):
    """
    Calculates centroids in the scVI latent space for each cluster in adata.
    Args:
        adata: AnnData object containing the scRNA-seq data with obsm['X_scVI'].
        cluster_key: Key in adata.obs indicating cluster assignments.
        embedding_key: Key in adata.obsm indicating the embedding to use (e.g., 'X_scVI').
        ndims: Number of dimensions in the embedding to consider.
    Returns:
        Value array of shape (num_clusters, ndims) with centroids for each cluster.
    """
    
    centroids = adata.obsm[embedding_key].copy()
    
    centroids_df = pd.DataFrame(centroids)
    centroids_df['cluster'] = adata.obs[cluster_key].values
    
    valid_clusters = []
    for cluster in adata.obs[cluster_key].cat.categories:
        if np.sum(adata.obs[cluster_key] == cluster) > 0:
            valid_clusters.append(cluster)
    
    if not valid_clusters:
        return np.zeros((0, ndims))
        
    centroids_df = centroids_df[centroids_df['cluster'].isin(valid_clusters)]
    centroids_df = centroids_df.groupby('cluster').mean()
    
    if np.isnan(centroids_df.values).any():
        centroids_df = centroids_df.dropna()
        
    return centroids_df.values
def Bayes_DE_Score(adata, cluster_1, cluster_2, min_pct, min_log2_fc, batch_size, model):
    """
    Calculates a score for differentially expressed genes between two clusters.
    Args:
        adata: AnnData object containing the scRNA-seq data with the specified embedding in obsm.
        cluster_1: First cluster label.
        cluster_2: Second cluster label.
        min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
        min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
        batch_size: Batch size for scVI differential expression.
        model: scVI model object for differential expression analysis.
    Returns:
        score: Sum of estimated log2 fold changes for genes passing the thresholds.
    """
    if model is None:
        print("Warning: No scVI model provided to Bayes_DE_Score. Returning high score to prevent merging.")
        return float('inf')
    
    scvi.settings.verbosity = logging.WARNING
    try:
        n_cells_1 = np.sum(adata.obs['leiden'] == cluster_1)
        n_cells_2 = np.sum(adata.obs['leiden'] == cluster_2)
        
        if n_cells_1 < 3 or n_cells_2 < 3:
            print(f"Not enough cells for DE: cluster {cluster_1}={n_cells_1} cells, cluster {cluster_2}={n_cells_2} cells")
            return float('inf')
        
        # Create a clean copy and fix any duplicate indices
        adata_subset = adata.copy()
        # Create mask for the two clusters
        mask = (adata_subset.obs['leiden'] == cluster_1) | (adata_subset.obs['leiden'] == cluster_2)
        adata_subset = adata_subset[mask].copy()
        # Ensure unique index after subsetting
        adata_subset.obs.index = pd.Index(np.arange(adata_subset.n_obs)).astype(str)
        
        # Ensure we have both clusters represented
        unique_clusters = adata_subset.obs['leiden'].unique()
        if len(unique_clusters) < 2:
            print(f"Only {len(unique_clusters)} cluster(s) found in subset")
            return float('inf')
        
        if cluster_1 not in unique_clusters or cluster_2 not in unique_clusters:
            print(f"Missing clusters in subset: {cluster_1} or {cluster_2}")
            return float('inf')
        
        # Perform differential expression
        de_genes = model.differential_expression(
            adata=adata_subset,
            mode='change',
            groupby='leiden', 
            group1=cluster_1, 
            group2=cluster_2,
            weights='importance', 
            batch_size=batch_size
        )
        
        # Ensure de_genes has a clean index
        de_genes = de_genes.reset_index(drop=True)
        
        # Filter genes based on criteria
        lfc_mask = abs(de_genes['lfc_mean']) > min_log2_fc
        pct_mask = (de_genes['non_zeros_proportion1'] > min_pct) | (de_genes['non_zeros_proportion2'] > min_pct)
        de_genes_filt = de_genes[lfc_mask & pct_mask].copy()
        
        if len(de_genes_filt) == 0:
            return 0.0
        
        # Filter by bayes factor and sum absolute log fold changes
        bayes_mask = de_genes_filt['bayes_factor'] > 3
        final_genes = de_genes_filt[bayes_mask]
        
        if len(final_genes) == 0:
            return 0.0
            
        return sum(abs(final_genes['lfc_mean']))
    except Exception as e:
        print(f"Error in Bayes_DE_Score: {e}")
        return float('inf')