from .detector3d_template import Detector3DTemplate
import torch.nn
from pcdet.models.backbones_3d.spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
class VoxelNeXt(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict,pruning=False):
        if pruning:
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
                if isinstance(cur_module,VoxelResBackBone8xVoxelNeXt):
                    break
            
            # loss=torch.mean((batch_dict['encoded_spconv_tensor']['features'] ** 2))
            loss=torch.mean((batch_dict['multi_scale_3d_features']['x_conv6'].features ** 2))
            return loss
            for pred_dict in self.dense_head.forward_ret_dict['pred_dicts']:
                loss+=torch.mean((pred_dict['center'] ** 2))+torch.mean((pred_dict['center_z'] ** 2))+torch.mean((pred_dict['dim'] ** 2))+torch.mean((pred_dict['rot'] ** 2))+torch.mean((pred_dict['hm'] ** 2))+torch.mean((pred_dict['vel'] ** 2))
            return loss
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)
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
        loss, tb_dict = self.dense_head.get_loss()
        
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
