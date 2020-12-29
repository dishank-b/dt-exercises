import torch
from ..utils import pairwise_iou, Boxes

def bayes_od_clustering(
        keep_inds,
        boxes_mean,
        boxes_variance,
        scores, 
        classes,
        iou_threshold=0.7):
    """
    Bayesian NMS clustering to output a single probability distribution per object in the scene.

    Args:
        boxes_mean: Nx4x1 tensor containing resulting means from mc_dropout. N is the number of anchors processed by the model.
        boxes_variance: Nx4x4 tensor containing resulting covariance matrices from mc_dropout.
        cat_count_res: NxC tensor containing the alpha counts from mc dropout. C is the number of categories.
        cluster_centers: Kx1 tensor containing the indices of centers of clusters based on minimum entropy. K is the number of clusters.
        iou_matrix: NxN matrix containing affinity measures between bounding boxes.
        affinity_threshold: scalar to determine the minimum affinity for clustering.
        
    Returns:
        final_scores: Kx4 tensor containing the parameters of the final categorical posterior distribution describing the objects' categories.
        final_means: Kx4x1 tensor containing the mean vectors of the posterior gaussian distribution describing objects' location in the scene.
        final_covs: Kx4x4 tensor containing the covariance matrices of the posterior gaussian distribution describing objects' location in the scene.
    """

    # Initialize lists to save per cluster results
    final_box_means = []
    final_box_covs = []
    final_box_scores = []
    final_box_classes = []
    
    iou_matrix = pairwise_iou(Boxes(boxes_mean), Boxes(boxes_mean))

    for cluster_center in keep_inds:

        # Get bounding boxes with affinity > threshold with the center of the
        # cluster
        cluster_mask = iou_matrix[:, cluster_center] > iou_threshold
        cluster_mask = cluster_mask & (classes==classes[cluster_center])
        cluster_means = boxes_mean[cluster_mask]
        cluster_covs = boxes_variance[cluster_mask]
        cluster_scores = scores[cluster_mask]

        # Compute mean and covariance of the final posterior distribution

        #inverse of variance
        var_inverses = 1/cluster_covs
        cluster_var = 1/torch.sum(var_inverses, dim=0) # 4x1
        final_box_covs.append(cluster_var)

        #calculating cluster mean
        cluster_mean = torch.sum(cluster_means*var_inverses, dim=0)*cluster_var
        final_box_means.append(cluster_mean)

        # calculating cluster class score - I just mean all the score but proper
        # way is using the dirchilet thing, as did below. So may be do that later
        final_box_scores.append(cluster_scores.mean())
        final_box_classes.append(classes[cluster_center])

        # # Compute the updated parameters of the categorical distribution
        # final_counts = predicted_boxes_class_counts[cluster_inds, :]
        # final_score = (final_counts) / \
        #     np.expand_dims(np.sum(final_counts, axis=1), axis=1)

        # if final_score.shape[0] > 3:
        #     cluster_center_score = np.expand_dims(predicted_boxes_class_counts[cluster_center, :] / np.sum(
        #         predicted_boxes_class_counts[cluster_center, :]), axis=0)
        #     cluster_center_score = np.repeat(
        #         cluster_center_score, final_score.shape[0], axis=0)

        #     cat_ent = entropy(cluster_center_score.T, final_score.T)

        #     inds = np.argpartition(cat_ent, 3)[:3]

        #     final_score = final_score[inds]
        #     final_counts = final_counts[inds]

        # final_score = np.mean(final_score, axis=0)
        # final_counts = np.sum(final_counts, axis=0)
        # final_box_scores.append(final_score)
        # final_box_classes.append(final_counts)
    try:
        final_box_means = torch.stack(final_box_means)
    except RuntimeError as error:
        print(len(keep_inds), len(boxes_mean))
    final_box_covs = torch.stack(final_box_covs)
    final_box_scores = torch.stack(final_box_scores)
    final_box_classes = torch.stack(final_box_classes)

    return final_box_means, final_box_covs, final_box_scores, final_box_classes