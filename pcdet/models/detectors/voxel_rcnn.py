from .detector3d_template import Detector3DTemplate
from pcdet.models.backbones_3d.spconv_backbone import VoxelBackBone8x
import torch
class VoxelRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict,pruning=False):
        if pruning:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
                if isinstance(cur_module,VoxelBackBone8x):
                    break
            loss=torch.mean((batch_dict['encoded_spconv_tensor'].features ** 2))
            return loss
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
        # if pruning:
        #     loss_rpn=torch.mean((self.dense_head.forward_ret_dict['cls_preds'] ** 2))+torch.mean((self.dense_head.forward_ret_dict['box_preds'] ** 2))+torch.mean((self.dense_head.forward_ret_dict['dir_cls_preds'] ** 2))
        #     loss_rcnn=torch.mean((self.roi_head.forward_ret_dict['rcnn_reg'] ** 2))+torch.mean((self.roi_head.forward_ret_dict['rcnn_cls'] ** 2))
        #     return loss_rpn+loss_rcnn
        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()
            
            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts
    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        
        loss_rpn, tb_dict = self.dense_head.get_loss()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)

        loss = loss + loss_rpn + loss_rcnn
        return loss, tb_dict, disp_dict
