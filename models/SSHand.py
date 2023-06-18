import torch
import torch.nn as nn
from torch.nn import functional as F

from models.backbones import build_backbone
from models.necks import build_neck
from loss.focalloss import QualityFocalLoss, DistributionFocalLoss
from loss.iouloss import GIoULoss
from utils.model_utils import ConvBnAct
from utils.weight_init import normal_init, bias_init_with_prob
from utils.utils import multi_apply
from utils.assigner import AlignOTAAssigner
from utils.dist import reduce_mean


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)

def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)


class Integral(nn.Module):
    """A fixed layer for calculating integral result from distribution.
    """
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))

    def forward(self, x):
        """Forward feature from the regression head to get integral result of
        bounding box location.
        """
        b, hw, _, _ = x.size()
        x = x.reshape(b * hw * 4, self.reg_max + 1)
        y = self.project.type_as(x).unsqueeze(1)
        x = torch.matmul(x, y).reshape(b, hw, 4)
        return x

class Scale(nn.Module):
    """A learnable scale parameter.
    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.
    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """
    def __init__(self, scale=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))

    def forward(self, x):
        return x * self.scale

class SSHand(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_num = cfg["model"]["classes"]
        self.backbone = build_backbone(cfg)
        self.neck = build_neck(cfg)

        self.head_num = 3
        self.head_len = 2
        # self.cls_convs = nn.ModuleList()
        # self.reg_convs = nn.ModuleList()
        # self.cls_layer = nn.ModuleList()
        # self.reg_layer = nn.ModuleList()

        self.strides = [8, 16, 32]

        # for i in range(self.head_num):
        #     cls_convs, reg_convs = self._build_not_shared_convs(256, 256)
        #     self.cls_convs.append(cls_convs)
        #     self.reg_convs.append(reg_convs)
        #     self.cls_layer.append(nn.Conv2d(256, self.cls_num, 3, 1, 1))
        #     self.reg_layer.append(nn.Conv2d(256, 4, 3, 1, 1))

        # self.assigner = AlignOTAAssigner(center_radius=2.5,
        #                                  cls_weight=1.0,
        #                                  iou_weight=3.0)
        # # self.cls_loss = QualityFocalLoss(use_sigmoid=False, beta=2.0, loss_weight=1.0)
        # # self.cls_loss = nn.BCEWithLogitsLoss()
        # self.cls_loss = FocalLoss(gamma=2.0)
        # self.reg_loss = GIoULoss(loss_weight=2.0)


        self.reg_max = 7
        self.assigner = AlignOTAAssigner(center_radius=2.5,
                                         cls_weight=1.0,
                                         iou_weight=3.0)
        self.integral = Integral(self.reg_max)
        self.loss_dfl = DistributionFocalLoss(loss_weight=0.25)
        self.loss_cls = QualityFocalLoss(use_sigmoid=False,
                                         beta=2.0,
                                         loss_weight=1.0)
        self.loss_bbox = GIoULoss(loss_weight=2.0)
        self.in_channels = [64, 64, 64]
        self.feat_channels  = self.in_channels
        self.cls_out_channels = self.cls_num
        self.last_kernel_size = 1
        self._init_layers()


    def init_bn(self, M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    def init_weights(self):
        """Initialize weights of the head."""
        for cls_conv in self.cls_convs:
            for m in cls_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        for reg_conv in self.reg_convs:
            for m in reg_conv:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        for i in range(len(self.strides)):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)


    def init_model(self):
        self.apply(self.init_bn)
        self.backbone.init_weights()
        self.neck.init_weights()
        self.init_weights()

    def _build_not_shared_convs(self, in_channel, feat_channels):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()

        # for i in range(self.head_len):
        #     chn = feat_channels if i > 0 else in_channel
        #     kernel_size = 3 if i > 0 else 1
        #     cls_convs.append(
        #         ConvBnAct(chn,
        #                   feat_channels,
        #                   kernel_size,
        #                   stride=1,
        #                   groups=1,
        #                   norm='bn',
        #                   act="silu"))
        #     reg_convs.append(
        #         ConvBnAct(chn,
        #                   feat_channels,
        #                   kernel_size,
        #                   stride=1,
        #                   groups=1,
        #                   norm='bn',
        #                   act="silu"))

        return cls_convs, reg_convs

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(len(self.strides)):
            cls_convs, reg_convs = self._build_not_shared_convs(
                self.in_channels[i], self.feat_channels[i])
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      self.cls_out_channels,
                      self.last_kernel_size, # 3
                      padding=self.last_kernel_size//2) for i in range(len(self.strides))
        ])

        self.gfl_reg = nn.ModuleList([
            nn.Conv2d(self.feat_channels[i],
                      4 * (self.reg_max + 1),
                      self.last_kernel_size, # 3
                      padding=self.last_kernel_size//2) for i in range(len(self.strides))
        ])

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def forward(self, x, label = None):   
        
        b1, b2, b3 = self.backbone(x)       
        
        f1, f2, f3 = self.neck([b1, b2, b3])        

        if self.training:
            in_features = [f1, f2, f3]
            return self.run_train(in_features, x, label)
        else:
            return self.run_eval(x)

    def run_train(self, features, images = None, labels = None):

        b, c, h, w = features[0].shape
        if labels is not None:
            gt_bbox_list = []
            gt_cls_list = []
            for label in labels:
                gt_bbox_list.append(label.bbox)
                gt_cls_list.append((label.get_field('labels')).long())

        # prepare priors for label assignment and bbox decode
        mlvl_priors_list = [
            self.get_single_level_center_priors(features[i].shape[0],
                                                features[i].shape[-2:],
                                                stride,
                                                dtype=torch.float32,
                                                device=images[0].device)
            for i, stride in enumerate(self.strides)
        ]
        mlvl_priors = torch.cat(mlvl_priors_list, dim=1)
        # forward for bboxes and classification prediction
        cls_scores, bbox_preds, bbox_before_softmax  = multi_apply(
            self.forward_single,
            features,
            self.cls_convs,
            self.reg_convs,
            self.gfl_cls,
            self.gfl_reg,
            self.scales,
        )
        cls_scores = torch.cat(cls_scores, dim=1)
        bbox_preds = torch.cat(bbox_preds, dim=1)
        bbox_before_softmax = torch.cat(bbox_before_softmax, dim=1)

        # calculating losses
        loss = self.loss(
            cls_scores,
            bbox_preds,
            bbox_before_softmax,
            gt_bbox_list,
            gt_cls_list,
            mlvl_priors,
        )
        return loss

    def forward_single(self, x, cls_convs, reg_convs, gfl_cls, gfl_reg, scale):
        """Forward feature of a single scale level.

        """
        
        cls_feat = x
        reg_feat = x

        for cls_conv, reg_conv in zip(cls_convs, reg_convs):
            cls_feat = cls_conv(cls_feat)
            reg_feat = reg_conv(reg_feat)

        bbox_pred = scale(gfl_reg(reg_feat)).float()
        cls_score = gfl_cls(cls_feat).sigmoid()

        N, C, H, W = bbox_pred.size()        
        
        if self.training:
            bbox_before_softmax = bbox_pred.reshape(N, 4, self.reg_max + 1, H,
                                                    W)
            bbox_before_softmax = bbox_before_softmax.flatten(
                start_dim=3).permute(0, 3, 1, 2)

            bbox_pred = F.softmax(bbox_pred.reshape(N, 4, self.reg_max + 1, H, W),
                                 dim=2)

            bbox_pred = bbox_pred.reshape(N, 4, self.reg_max + 1, H, W)

            cls_score = cls_score.flatten(start_dim=2).permute(
                0, 2, 1)  # N, h*w, self.num_classes+1
            bbox_pred = bbox_pred.flatten(start_dim=3).permute(
                0, 3, 1, 2)  # N, h*w, 4, self.reg_max+1

        if self.training:
            return cls_score, bbox_pred, bbox_before_softmax
        else:
            return cls_score, bbox_pred

    def get_single_level_center_priors(self, batch_size, featmap_size, stride,
                                       dtype, device):

        h, w = featmap_size
        x_range = (torch.arange(0, int(w), dtype=dtype,
                                device=device)) * stride
        y_range = (torch.arange(0, int(h), dtype=dtype,
                                device=device)) * stride

        x = x_range.repeat(h, 1)
        y = y_range.unsqueeze(-1).repeat(1, w)

        y = y.flatten()
        x = x.flatten()
        strides = x.new_full((x.shape[0], ), stride)
        priors = torch.stack([x, y, strides, strides], dim=-1)

        return priors.unsqueeze(0).repeat(batch_size, 1, 1)

    def loss(
        self,
        cls_scores,
        bbox_preds,
        bbox_before_softmax,
        gt_bboxes,
        gt_labels,
        mlvl_center_priors,
        gt_bboxes_ignore=None,
    ):
        """Compute losses of the head.

        """
        device = cls_scores[0].device
        

        # get decoded bboxes for label assignment
        dis_preds = self.integral(bbox_preds) * mlvl_center_priors[..., 2,
                                                                   None]
        decoded_bboxes = distance2bbox(mlvl_center_priors[..., :2], dis_preds)
        cls_reg_targets = self.get_targets(cls_scores,
                                           decoded_bboxes,
                                           gt_bboxes,
                                           mlvl_center_priors,
                                           gt_labels_list=gt_labels)

        if cls_reg_targets is None:
            return None

        (labels_list, label_scores_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, dfl_targets_list, num_pos) = cls_reg_targets

        num_total_pos = max(
            reduce_mean(torch.tensor(num_pos).type(
                torch.float).to(device)).item(), 1.0)
        
        
        labels = torch.cat(labels_list, dim=0)
        label_scores = torch.cat(label_scores_list, dim=0)
        bbox_targets = torch.cat(bbox_targets_list, dim=0)
        dfl_targets = torch.cat(dfl_targets_list, dim=0)

        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # bbox_preds = bbox_preds.reshape(-1, 4 * (self.reg_max + 1))
        bbox_before_softmax = bbox_before_softmax.reshape(
            -1, 4 * (self.reg_max + 1))
        decoded_bboxes = decoded_bboxes.reshape(-1, 4)
        loss_qfl = self.loss_cls(cls_scores, (labels, label_scores),
                                 avg_factor=num_total_pos)

        pos_inds = torch.nonzero((labels >= 0) & (labels < self.cls_num),
                                 as_tuple=False).squeeze(1)

        weight_targets = cls_scores.detach()
        weight_targets = weight_targets.max(dim=1)[0][pos_inds]
        norm_factor = max(reduce_mean(weight_targets.sum()).item(), 1.0)

        if len(pos_inds) > 0:
            loss_bbox = self.loss_bbox(
                decoded_bboxes[pos_inds],
                bbox_targets[pos_inds],
                weight=weight_targets,
                avg_factor=1.0 * norm_factor,
            )            
            # loss_dfl = self.loss_dfl(
            #     bbox_before_softmax[pos_inds].reshape(-1, self.reg_max + 1),
            #     dfl_targets[pos_inds].reshape(-1),
            #     weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
            #     avg_factor=4.0 * norm_factor,
            # )
            # loss_bbox = bbox_preds.sum() / norm_factor * 0.0
            loss_dfl = bbox_preds.sum() / norm_factor * 0.0
        else:
            loss_bbox = bbox_preds.sum() / norm_factor * 0.0
            loss_dfl = bbox_preds.sum() / norm_factor * 0.0
            # logger.info(f'No Positive Samples on {bbox_preds.device}! May cause performance decrease. loss_bbox:{loss_bbox:.3f}, loss_dfl:{loss_dfl:.3f}, loss_qfl:{loss_qfl:.3f} ')

        
        total_loss = loss_qfl + loss_bbox + loss_dfl

        return dict(
            total_loss=total_loss,
            loss_cls=loss_qfl,
            loss_bbox=loss_bbox,
            loss_dfl=loss_dfl,
        )

    def get_targets(self,
                    cls_scores,
                    bbox_preds,
                    gt_bboxes_list,
                    mlvl_center_priors,
                    gt_labels_list=None,
                    unmap_outputs=True):
        """Get targets for GFL head.

        """
        num_imgs = mlvl_center_priors.shape[0]

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        (all_labels, all_label_scores, all_label_weights, all_bbox_targets,
         all_bbox_weights, all_dfl_targets, all_pos_num) = multi_apply(
             self.get_target_single,
             mlvl_center_priors,
             cls_scores,
             bbox_preds,
             gt_bboxes_list,
             gt_labels_list,
         )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        all_pos_num = sum(all_pos_num)

        return (all_labels, all_label_scores, all_label_weights,
                all_bbox_targets, all_bbox_weights, all_dfl_targets,
                all_pos_num)

    def get_target_single(self,
                          center_priors,
                          cls_scores,
                          bbox_preds,
                          gt_bboxes,
                          gt_labels,
                          unmap_outputs=True,
                          gt_bboxes_ignore=None):
        """Compute regression, classification targets for anchors in a single
        image.

        """
        # assign gt and sample anchors

        num_valid_center = center_priors.shape[0]

        labels = center_priors.new_full((num_valid_center, ),
                                        self.cls_num,
                                        dtype=torch.long)
        label_weights = center_priors.new_zeros(num_valid_center,
                                                dtype=torch.float)
        label_scores = center_priors.new_zeros(num_valid_center,
                                               dtype=torch.float)

        bbox_targets = torch.zeros_like(center_priors)
        bbox_weights = torch.zeros_like(center_priors)
        dfl_targets = torch.zeros_like(center_priors)

        if gt_labels.size(0) == 0:

            return (labels, label_scores, label_weights, bbox_targets,
                    bbox_weights, dfl_targets, 0)

        assign_result = self.assigner.assign(cls_scores.detach(),
                                             center_priors,
                                             bbox_preds.detach(), gt_bboxes,
                                             gt_labels)

        pos_inds, neg_inds, pos_bbox_targets, pos_assign_gt_inds = self.sample(
            assign_result, gt_bboxes)
        pos_ious = assign_result.max_overlaps[pos_inds]

        if len(pos_inds) > 0:
            labels[pos_inds] = gt_labels[pos_assign_gt_inds]
            label_scores[pos_inds] = pos_ious
            label_weights[pos_inds] = 1.0

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            dfl_targets[pos_inds, :] = (bbox2distance(
                center_priors[pos_inds, :2] / center_priors[pos_inds, None, 2],
                pos_bbox_targets / center_priors[pos_inds, None, 2]))
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        # map up to original set of anchors

        return (labels, label_scores, label_weights, bbox_targets,
                bbox_weights, dfl_targets, pos_inds.size(0))

    def sample(self, assign_result, gt_bboxes):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0,
                                 as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0,
                                 as_tuple=False).squeeze(-1).unique()
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]

        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds