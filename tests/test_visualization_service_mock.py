import unittest
from unittest.mock import MagicMock, patch
import logging
import numpy as np
from pathlib import Path
import sys
import tempfile
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.visualization_service import SkeletonVisualizationService
from src.pose_extract.track_solution.analysis.analysis_results import AnalysisResult

class TestSkeletonVisualizationService(unittest.TestCase):
    
    def setUp(self):
        # Create a temporary directory for outputs
        self.test_dir = tempfile.mkdtemp()
        self.config = {
            "output_dir": self.test_dir,
            "draw_options": {
                "show_interpolated": True
            },
            "video_writer": {
                "fps": 30,
                "codec": "mp4v"
            },
            "overwrite": True
        }
        self.service = SkeletonVisualizationService(self.config)
        
        # Mock VideoSource
        self.mock_video_source_patchless = MagicMock()
        self.mock_video_source_patchless.target_width = 100
        self.mock_video_source_patchless.target_height = 100
        
        # Mock frame
        self.mock_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.mock_video_source_patchless.read.return_value = (True, self.mock_frame)
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("src.core.visualization_service.VideoSource")
    @patch("src.core.visualization_service.cv2.VideoWriter")
    def test_visualize_analysis_segments(self, mock_video_writer, mock_video_source_cls):
        # Setup Mocks
        mock_source_instance = mock_video_source_cls.return_value
        mock_source_instance.target_width = 100
        mock_source_instance.target_height = 100
        mock_source_instance.read.return_value = (True, self.mock_frame)
        
        mock_writer_instance = mock_video_writer.return_value
        mock_writer_instance.isOpened.return_value = True

        # Setup TrackManager Mock
        mock_track_manager = MagicMock()
        mock_track = MagicMock()
        mock_track.track_id = 1
        mock_track_manager.get_track.return_value = mock_track

        # Setup AnalysisResult
        analysis_result = AnalysisResult(track_id=1)
        # Add a fake segment: type="walking", start=0, end=10
        analysis_result.add_segment_result("walking", None, [(0, 10)])
        
        analysis_results = {1: analysis_result}
        
        video_info = {
            "video_path": "dummy_video.mp4",
            "video_name": "dummy_video",
            "fps": 30,
            "width": 100,
            "height": 100
        }
        
        # Mock Path.exists to return True
        with patch("pathlib.Path.exists", return_value=True):
            # Execute
            generated = self.service.visualize_analysis_segments(
                track_manager=mock_track_manager,
                analysis_results=analysis_results,
                video_info=video_info,
                target_segment_type="walking"
            )
            
        # Assertions
        # 1. Check if VideoSource was initialized
        mock_video_source_cls.assert_called_with("dummy_video.mp4")
        mock_source_instance.open.assert_called_once()
        
        # 2. Check if VideoWriter was called for the segment
        # file name format: {video_stem}_T{t_id}_{s_type}_{s_idx:03d}_F{start_f}-{end_f}.mp4
        # dummy_video_T1_walking_000_F0-10.mp4
        expected_filename = Path(self.test_dir) / "dummy_video" / "dummy_video_T1_walking_000_F0-10.mp4"
        # We check if writer was initialized at least once
        self.assertTrue(mock_video_writer.called)
        
        # 3. Check if frames were read and written
        # Range 0 to 10 inclusive = 11 frames
        # VideoWriter.write should be called 11 times
        self.assertEqual(mock_writer_instance.write.call_count, 11)
        
        # 4. Check return value
        self.assertEqual(len(generated), 1)
        self.assertIn(str(expected_filename), generated[0])

    @patch("src.core.visualization_service.VideoSource")
    def test_visualize_no_segments(self, mock_video_source_cls):
         # Setup TrackManager Mock
        mock_track_manager = MagicMock()
        
        # Empty AnalysisResult
        analysis_results = {1: AnalysisResult(track_id=1)}
        
        video_info = {
            "video_path": "dummy_video.mp4",
            "video_name": "dummy_video"
        }
        
        with patch("pathlib.Path.exists", return_value=True):
            generated = self.service.visualize_analysis_segments(
                track_manager=mock_track_manager,
                analysis_results=analysis_results,
                video_info=video_info,
                target_segment_type="walking"
            )
            
        self.assertEqual(len(generated), 0)
        mock_video_source_cls.assert_not_called()

if __name__ == "__main__":
    unittest.main()
