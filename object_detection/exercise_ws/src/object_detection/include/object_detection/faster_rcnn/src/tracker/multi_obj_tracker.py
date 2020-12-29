from __future__ import absolute_import
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment as assignment

from .kalman_filter import KalmanFilter
from . import linear_assignment
from . import iou_matching
from .track import Track

from ..utils import Boxes, pairwise_iou, Matcher, Instances


class MultiObjTracker(object):
    """
    This is the multi-target tracker.
    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    """
    def __init__(self, cfg):
        self.max_iou_distance = cfg.TRACKING.IOU_DISTANCE
        self.max_age = cfg.TRACKING.MAX_AGE
        self.n_init = cfg.TRACKING.N_INIT
        std_weight_position = cfg.TRACKING.POS_NOISE_WEIGHT
        std_weight_velocity = cfg.TRACKING.VEL_NOISE_WEIGHT

        self.kf = KalmanFilter(std_weight_position, std_weight_velocity)
        self.tracks_matcher = Matcher(
            cfg.ROI_HEADS.IOU_THRESHOLDS,
            cfg.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        self.batch_size = cfg.TRAIN.BATCH_SIZE
        # Each element of self.tracks is list of all tracked object of that sequence.
        # Each batch index correspond to one sequence.

        self.tracks = [[] for _ in range(self.batch_size)]
        self._next_id = [1 for _ in range(self.batch_size)]

    def reinit_state(self):
        # Each element of self.tracks is list of all tracked object of that sequence.
        # Each batch index correspond to one sequence.
        for x in self.tracks:
            x.clear()

        self._next_id = [1 for _ in range(self.batch_size)]

    # def forward(self, detections, gt_target, is_training):
    def __call__(self, detections, gt_target, is_training):

        for idx, detect in enumerate(detections):
            boxes = detect.pred_boxes.tensor
            variance = detect.pred_variance
            self._predict(idx)
            # if
            _ = [x.draw("./logs/testing/results", "predict") for x in self.tracks_to_instance(gt_target)]
            self._update(boxes, variance, idx)

        tracks = self.tracks_to_instance(gt_target)
        # _ = [x.draw("./logs/testing/results", "update") for x in tracks]

        if is_training:
            loss = self.loss(tracks, gt_target)
            return tracks, {"track_loss": loss}

        return tracks, {}

    def tracks_to_instance(self, targets):
        batch_tracks = []
        for tracks, target in zip(self.tracks, targets):

            #When there is not tracks.
            if len(tracks) != 0:
                track_boxes = Boxes(torch.stack([x.box for x in tracks],
                                                dim=0))
                track_var = torch.stack([x.box_var for x in tracks], dim=0)
            else:
                track_boxes = Boxes(torch.Tensor())
                track_var = torch.Tensor()

            tracks = Instances(image_size=target.image_size,
                               image_path=target.image_path,
                               pred_boxes=track_boxes,
                               pred_variance=track_var)

            batch_tracks.append(tracks)

        return batch_tracks

    def loss(self, tracks, gt_targets):
        """
                Loss attenuation implementation
        Returns:
            scalar Tensor
        """
        tracks_box_list = []
        tracks_var_list = []
        target_box_list = []

        for track, target in zip(tracks, gt_targets):

            # When in cases there is not tracks to calculate the loss
            # we can add the target to the tracks, where loss will be
            # zero and no gradient will flow and we wont get any error.
            if len(track) == 0:
                track = Instances(image_size=target.image_size,
                                  image_path=target.image_path,
                                  pred_boxes=target.gt_boxes,
                                  pred_variance=torch.ones(
                                      len(target),
                                      4,
                                      device=torch.device("cuda")))

            else:
                match_quality_matrix = pairwise_iou(target.gt_boxes,
                                                    track.pred_boxes)

                matched_idxs, matched_labels = self.tracks_matcher(
                    match_quality_matrix)
                fg_inds = torch.nonzero(matched_labels == 1).squeeze(1)

                target = target[matched_idxs]

                track = track[fg_inds]
                target = target[fg_inds]

                # Used in drawing associated gt_boxes
                track.gt_boxes = target.gt_boxes

            tracks_box_list.append(track.pred_boxes)
            tracks_var_list.append(track.pred_variance)
            target_box_list.append(target.gt_boxes)

            # track.draw("./logs/backpropkf/results")

        tracks_var_batch = torch.cat(tracks_var_list)
        tracks_boxes_batch = Boxes.cat(tracks_box_list)
        target_boxes_batch = Boxes.cat(target_box_list)

        # print("Track Boxes:", tracks_boxes_batch)
        # print("Target Boxes:", target_boxes_batch)
        # print("Variance:", tracks_var_batch)
        ## Computing the loss attenuation
        mse = (tracks_boxes_batch.tensor - target_boxes_batch.tensor)**2
        # print(mse, tracks_var_batch)
        loss_attenuation_final = (mse / tracks_var_batch +
                                  torch.log(tracks_var_batch)).mean()

        # weighting scaling by 50.0.
        return loss_attenuation_final

    def _predict(self, idx):
        """Propagate track state distributions one time step forward.
        This function should be called once every time step, before `update`.
        """
        for track in self.tracks[idx]:
            track.predict(self.kf)

    def _update(self, detections, measurement_var, idx):
        """Perform measurement update and track management.
        Parameters
        ----------
        detections : List[N,4]
            A list of detections at the current time step.
        measurement_var :
            A list of measurement noise of each bounding box at current time step.
        """

        # Run matching cascade.
        # matches, unmatched_tracks, unmatched_detections = \
        #     self._match(detections)


        matches, unmatched_tracks, unmatched_detections = \
            self._associate_detections_to_trackers(detections, self.tracks[idx])

        # print("Update state print:")
        # print(matches, unmatched_tracks, unmatched_detections)
        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[idx][track_idx].update(self.kf,
                                               detections[detection_idx],
                                               measurement_var[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[idx][track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx],
                                 measurement_var[detection_idx], idx)

        self.tracks[idx] = [t for t in self.tracks[idx] if not t.is_deleted()]

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            cost_matrix = iou_matching.iou_cost(tracks, dets, track_indices,
                                                detection_indices)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks_idx = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()
        ]
        unconfirmed_tracks_idx = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()
        ]

        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.max_iou_distance, self.max_age,
                self.tracks, detections, confirmed_tracks_idx)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks_idx + [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a
            if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        return matches, unmatched_tracks, unmatched_detections

    def _associate_detections_to_trackers(self,
                                          detections,
                                          trackers,
                                          iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """

        if (len(trackers) == 0):
            return np.empty((0, 2), dtype=int), np.empty(
                (0, 5), dtype=int), np.arange(len(detections))

        iou_matrix = np.zeros((len(detections), len(trackers)),
                              dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det, trk.box)

        matched_indices = assignment(-iou_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if (d not in matched_indices[:, 0]):
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if (t not in matched_indices[:, 1]):
                unmatched_trackers.append(t)

        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if (iou_matrix[m[0], m[1]] < iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))
        if (len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        matches = matches[:, [1,
                              0]]  # 0th col is tracking, 1st col is detection

        return matches, np.array(unmatched_trackers), np.array(
            unmatched_detections)

    def _initiate_track(self, detection, detection_noise, idx):
        mean, covariance = self.kf.initiate(detection, detection_noise)
        # print("initiate tracks", mean, covariance)
        self.tracks[idx].append(
            Track(mean, covariance, self._next_id[idx], self.n_init,
                  self.max_age))
        self._next_id[idx] += 1

    def _iou(self, bb_test, bb_gt):
        """
        Computes IUO between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_test = bb_test.detach().cpu().numpy()
        bb_gt = bb_gt.detach().cpu().numpy()

        xx1 = np.maximum(bb_test[0], bb_gt[0])
        yy1 = np.maximum(bb_test[1], bb_gt[1])
        xx2 = np.minimum(bb_test[2], bb_gt[2])
        yy2 = np.minimum(bb_test[3], bb_gt[3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        wh = w * h
        o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                  (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
        return (o)
