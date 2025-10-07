import scanpy as sc
import scvi
import leidenalg
import numpy as np
import igraph
import pandas as pd
import logging
def Iterative_Clustering_scVI(adata, ndims=30, num_iterations=20, min_pct=0.4, min_log2_fc=2, batch_size=2048, min_bayes_score=8, min_cluster_size=4, model=None):
    """
    Wrapper function to perform iterative clustering using scVI and Leiden algorithm.
    Args:
        adata: AnnData object containing the scRNA-seq data with obsm['X_scVI'].
        ndims: Number of scVI latent dimensions to use.
        num_iterations: Maximum number of clustering iterations.
        min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
        min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
        batch_size: Batch size for scVI differential expression.
        min_bayes_score: Minimum score for a gene to be considered differentially expressed.
        min_cluster_size: Minimum size of clusters to retain.
    Returns:
        adata: AnnData object with updated clustering in adata.obs['leiden'].
    """
    adata.obs['leiden']='1'
    adata.obs['leiden'] = adata.obs['leiden'].astype('category')
    previous_num_clusters = 1
    for i in range(num_iterations):
        adata = Clustering_Iteration(adata, ndims=ndims, min_pct=min_pct, min_log2_fc=min_log2_fc, batch_size=batch_size, min_bayes_score=min_bayes_score, model=model)
        if len(adata.obs['leiden'].cat.categories) == previous_num_clusters:
            break
        previous_num_clusters = len(adata.obs['leiden'].cat.categories)
    return adata
def Clustering_Iteration(adata, ndims=30, min_pct=0.4, min_log2_fc=2, batch_size=2048, min_bayes_score=8, min_cluster_size=4, model=None):
    """
    Performs one iteration of clustering and merging.
    Args:
         adata: AnnData object containing the scRNA-seq data with obsm['X_scVI'].
         ndims: Number of scVI latent dimensions to use.
         min_pct: Minimum percentage of cells expressing a gene to consider it for differential expression.
         min_log2_fc: Minimum log2 fold change for a gene to be considered differentially expressed.
         batch_size: Batch size for scVI differential expression.
         min_bayes_score: Minimum score for a gene to be considered differentially expressed.
         min_cluster_size: Minimum size of clusters to retain.
         model: scVI model object for differential expression analysis. If None, clustering will still occur but differential expression scoring will be skipped.
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
            sc.pp.neighbors(cluster_adata, use_rep='X_scVI', n_neighbors=int(np.floor(cluster_adata.n_obs/2)), n_pcs=ndims)
        else:
            sc.pp.neighbors(cluster_adata, use_rep='X_scVI', n_pcs=ndims)
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
            centroids = Find_Centroids(cluster_adata, cluster_key='leiden', embedding_key='X_scVI', ndims=ndims)
            
            if centroids.shape[0] < 2:
                break
                
            centroid_map = {subcluster: i for i, subcluster in enumerate(nonempty_sub_clusters)}
            
            from sklearn.metrics import pairwise_distances
            dist_matrix = pairwise_distances(centroids)
            np.fill_diagonal(dist_matrix, np.inf)
            
            min_distance = np.inf
            closest_pair = None
            
            for sub_cluster in nonempty_sub_clusters:
                if sub_cluster not in centroid_map:
                    continue
                    
                idx = centroid_map[sub_cluster]
                
                if idx >= len(nonempty_sub_clusters) or idx >= dist_matrix.shape[0]:
                    continue
                    
                distances = dist_matrix[idx]
                
                if np.all(np.isinf(distances)):
                    continue
                    
                closest_idx = np.argmin(distances)
                
                if closest_idx == idx or distances[closest_idx] == np.inf:
                    continue
                
                if closest_idx >= len(nonempty_sub_clusters):
                    continue
                    
                closest_sub_cluster = nonempty_sub_clusters[closest_idx]
                
                if (sub_cluster, str(closest_sub_cluster)) in merged_pairs or (str(closest_sub_cluster), sub_cluster) in merged_pairs:
                    continue
                
                if distances[closest_idx] < min_distance:
                    min_distance = distances[closest_idx]
                    closest_pair = (sub_cluster, closest_sub_cluster)
            
            if closest_pair is None:
                break
                
            sub_cluster, closest_sub_cluster = closest_pair
                
            n_cells_sub = np.sum(cluster_adata.obs['leiden'] == sub_cluster)
            n_cells_closest = np.sum(cluster_adata.obs['leiden'] == closest_sub_cluster)
            
            if n_cells_sub < 3 or n_cells_closest < 3:  # Need at least 3 cells for DE
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                continue
                
            bayes_de_score = Bayes_DE_Score(cluster_adata, sub_cluster, closest_sub_cluster, min_pct, min_log2_fc, batch_size, model=model)
            
            if bayes_de_score < min_bayes_score:
                cluster_adata.obs.loc[cluster_adata.obs['leiden'] == closest_sub_cluster, 'leiden'] = sub_cluster
                merged_pairs.append((sub_cluster, str(closest_sub_cluster)))
                changes_made = True
        
        cluster_adata.obs['leiden'] = cluster_adata.obs['leiden'].cat.remove_unused_categories()
        
        final_sub_clusters = cluster_adata.obs['leiden'].cat.categories
        final_nonempty_sub_clusters = [subcluster for subcluster in final_sub_clusters if np.sum(cluster_adata.obs['leiden'] == subcluster) > 0]
        
        if len(final_nonempty_sub_clusters) <= 1:
            continue
            
        new_labels = []
        for i, subcluster in enumerate(final_nonempty_sub_clusters, 1):
            new_label = f"{cluster}_{i}"
            new_labels.append(new_label)
            
        for new_label in new_labels:
            if new_label not in adata.obs['leiden'].cat.categories:
                adata.obs['leiden'] = adata.obs['leiden'].cat.add_categories([new_label])
        
        for i, subcluster in enumerate(final_nonempty_sub_clusters):
            new_label = new_labels[i]
            subcluster_mask = cluster_adata.obs['leiden'] == subcluster
            original_indices = cluster_adata.obs.index[subcluster_mask]
            adata.obs.loc[original_indices, 'leiden'] = new_label
    
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
        adata: AnnData object containing the scRNA-seq data with obsm['X_scVI'].
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
        
        # Reset index to avoid duplicate index issues
        adata_subset.obs = adata_subset.obs.reset_index(drop=True)
        adata_subset.obs.index = adata_subset.obs.index.astype(str)
        
        # Create mask for the two clusters
        mask = (adata_subset.obs['leiden'] == cluster_1) | (adata_subset.obs['leiden'] == cluster_2)
        adata_subset = adata_subset[mask].copy()
        
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
        
        # Filter genes based on criteria
        de_genes_filt = de_genes[
            (abs(de_genes['lfc_mean']) > min_log2_fc) & 
            ((de_genes['non_zeros_proportion1'] > min_pct) | (de_genes['non_zeros_proportion2'] > min_pct))
        ]
        
        return sum(abs(de_genes_filt['lfc_mean'])[de_genes_filt['bayes_factor'] > 3])
        
    except Exception as e:
        print(f"Error in Bayes_DE_Score: {e}")
        return float('inf')