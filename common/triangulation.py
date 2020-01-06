import torch
import numpy as np
from common.camera import normalize_screen_coordinates_multiview, image_coordinates_multiview


def homogeneous_to_euclidean(points):
    """Converts homogeneous points to euclidean

    Args:
        points numpy array or torch tensor of shape (N, M + 1): N homogeneous points of dimension M

    Returns:
        numpy array or torch tensor of shape (N, M): euclidean points
    """
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    elif torch.is_tensor(points):
        return (points.transpose(1, 0)[:-1] / points.transpose(1, 0)[-1]).transpose(1, 0)
    else:
        raise TypeError("Works only with numpy arrays and PyTorch tensors.")


def triangulate_point_from_multiple_views_linear_torch(proj_matricies, points, confidences=None):
    """Similar as triangulate_point_from_multiple_views_linear() but for PyTorch.
    For more information see its documentation.
    Args:
        proj_matricies torch tensor of shape (N, 3, 4): sequence of projection matricies (3x4)
        points torch tensor of of shape (N, 2): sequence of points' coordinates
        confidences None or torch tensor of shape (N,): confidences of points [0.0, 1.0].
                                                        If None, all confidences are supposed to be 1.0
    Returns:
        point_3d numpy torch tensor of shape (3,): triangulated point
    """
    assert len(proj_matricies) == len(points)

    n_views = len(proj_matricies)

    if confidences is None:
        confidences = torch.ones(n_views, dtype=torch.float32, device=points.device)

    A = proj_matricies[:, 2:3].expand(n_views, 2, 4) * points.view(n_views, 2, 1)
    A -= proj_matricies[:, :2]
    A *= confidences.view(-1, 1, 1)

    u, s, vh = torch.svd(A.contiguous().view(-1, 4))

    point_3d_homo = -vh[:, 3]
    point_3d = homogeneous_to_euclidean(point_3d_homo.unsqueeze(0))[0]

    return point_3d


def L2acc(triangulate_3d, target_3d, topk_points=18):
    return torch.mean(torch.norm(triangulate_3d[:, 0:topk_points] - target_3d[:, 0:topk_points], dim=2))


def triangulation_acc(preds, data_dict, all_metric=False):
    device = preds.device
    N, J, V = preds.shape
    target_3d = data_dict['target_3d']
    target_score = data_dict['target_score']
    input = data_dict['2d_ori'].permute(0,3,2,1)
    cameras = data_dict['camera']

    ltntri_output_after = torch.zeros(N,J,3).to(device)
    ltntri_output_svd = torch.zeros_like(ltntri_output_after).to(device)
    ltntri_output_before = torch.zeros_like(ltntri_output_after).to(device)
    best_output = torch.zeros_like(ltntri_output_after).to(device)

    for n in range(N):
        for j in range(J):
            input_p = input[n,0:2,j,:].view(-1,V).permute(1,0)
            pred_p = preds[n,j,:].view(V)
            input_p_c = input[n,2,j,:].view(V)
            best_p_c = target_score[n,j,:].view(V)
            ltntri_output_after[n,j,:] = triangulate_point_from_multiple_views_linear_torch(cameras[n], input_p, pred_p)
            if all_metric:
                ltntri_output_before[n,j,:] = triangulate_point_from_multiple_views_linear_torch(cameras[n], input_p, input_p_c)
                best_output[n,j,:] = triangulate_point_from_multiple_views_linear_torch(cameras[n], input_p, best_p_c)
                ltntri_output_svd[n,j,:] = triangulate_point_from_multiple_views_linear_torch(cameras[n], input_p)

    if all_metric:
        output_3d_dict = {
            "ltr_before": ltntri_output_before,
            "ltr_after":ltntri_output_after,
            "ltr_best":best_output,
            "ltr_svd":ltntri_output_svd,
        }
    else:
        output_3d_dict = {
            "ltr_after": ltntri_output_after
        }

    return output_3d_dict, target_3d


    # ltntri_acc_before = L2acc(ltntri_output_before, target_3d, 18)
    # ltntri_acc_after = L2acc(ltntri_output_after, target_3d, 18)
    # ltntri_acc_svd = L2acc(ltntri_output_svd, target_3d, 18)
    # best_acc = L2acc(best_output, target_3d, 18)
    #
    # if all_metric:
    #     acc_dict = {
    #         "ltr_before": ltntri_acc_before,
    #         "ltr_after":ltntri_acc_after,
    #         "ltr_best":best_acc,
    #         "ltr_svd":ltntri_acc_svd,
    #     }
    # else:
    #     acc_dict = {
    #         "ltr_after": ltntri_acc_after
    #     }
    #
    # return acc_dict, ltntri_output_after, target_3d
