from config.config import ConfigManager
from vendor.BoTSORT.tracker.bot_sort import BoTSORT

# class BoTSORTArgs:
#     def __init__(self, **kwargs):
#         self.track_high_thresh = 0.6
#         self.track_low_thresh = 0.1
#         self.new_track_thresh = 0.7
#         self.track_buffer = 30
#         self.with_reid = False
#         self.match_thresh = 0.8
#         self.proximity_thresh = 0.5
#         self.appearance_thresh = 0.25
#         self.cmc_method = 'none'
#         self.name = 'exp'
#         self.ablation = False
#         self.mot20 = False
#         self.fast_reid_config = ""
#         self.fast_reid_weights = ""
#         self.device = 'cuda'
class BoTSORTArgs:
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)

class BoTSORTWrapper:
    def __init__(self, config):
        # config = ConfigManager.get_config()['botsort']
        args = BoTSORTArgs(config)
        self.tracker = BoTSORT(args)

    def update(self, img, bboxes, scores, keypoints=None, classes=None):
        """
        Update the tracker with new detections
         
        Args:
            detections: Detection results (format depends on what BoSORT expects)
            frame: Optional current frame for visualization
             
        Returns:
            Tracking results
        """
        return self.tracker.update(img, bboxes, scores, keypoints, classes)
        
    def close(self):
        """清理資源"""
        # BoTSORT 目前沒有需要特殊清理的資源
        pass
