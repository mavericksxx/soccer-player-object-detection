# PLAYER TRACKER

# this script implements a comprehensive player tracking system for soccer videos that:
# - detects players using a finetuned YOLOv11 model (best.pt)
# - assigns unique player ids and tracks across frames
# - handles player re identification when they exit and re enter the frame

# FeatureExtractor class extracts color histogram features from HSV color space (32 bins per channel),
# calculates spatial features (normalized center coordinates, width, height), combines visual and spatial 
# features for robust player identification uses normalized histograms and bounded bounding box extraction.

# PlayerTracker class initializes players with unique ids, feature history, and tracking metadata. 
# it detects players using YOLO with configurable confidence threshold (default 0.5) and implements 
# a matching algorithm using spatial distance and visual similarity. it also maintains feature history 
# (last 5 frames) and bbox history (last 10 frames) per player and uses adaptive similarity thresholds 
# based on how recently a player was seen. it uses a combined similarity scoring (70% spatial weight, 
# 30% visual feature weight) andsupports up to 100 frames of player absence before considering them for cleanup.


import cv2
import numpy as np
from ultralytics import YOLO
import torch
from collections import defaultdict, deque
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import Dict, List, Tuple, Optional
import time


class FeatureExtractor:
    def __init__(self):
        self.color_bins = 32  
        
    def extract_color_histogram(self, image_patch: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image_patch, cv2.COLOR_BGR2HSV)

        hist_h = cv2.calcHist([hsv], [0], None, [self.color_bins], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [self.color_bins], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [self.color_bins], [0, 256])

        features = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()])
        features = features / (np.sum(features) + 1e-6)  # Normalize
        
        return features
    
    def extract_spatial_features(self, bbox: Tuple[int, int, int, int], 
                                frame_shape: Tuple[int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        frame_h, frame_w = frame_shape[:2]
        
        center_x = (x1 + x2) / (2 * frame_w)
        center_y = (y1 + y2) / (2 * frame_h)
        width = (x2 - x1) / frame_w
        height = (y2 - y1) / frame_h
        
        return np.array([center_x, center_y, width, height])
    
    def extract_features(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1], x2)
        y2 = min(image.shape[0], y2)
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.color_bins * 3 + 4)

        patch = image[y1:y2, x1:x2]

        color_features = self.extract_color_histogram(patch)
        spatial_features = self.extract_spatial_features(bbox, image.shape)

        combined_features = np.concatenate([color_features, spatial_features])
        
        return combined_features


class PlayerTracker:
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.5):

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.feature_extractor = FeatureExtractor()

        self.players = {} 
        self.next_player_id = 1
        self.frame_count = 0
        self.initialization_frames = 90

        self.similarity_threshold = 0.05  
        self.max_missing_frames = 100 
        self.feature_history_size = 5  
        self.spatial_weight = 0.7   
        self.feature_weight = 0.3   
        self.max_distance_threshold = 0.5  

        self.recent_threshold = 0.08
        self.old_threshold = 0.05  
        self.recent_frame_limit = 25  

        self.detection_times = []
        self.tracking_times = []

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        start_time = time.time()
        
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if class_id in [1, 2]:  
                        detections.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id
                        })
        
        detection_time = time.time() - start_time
        self.detection_times.append(detection_time)
        
        return detections
    
    def calculate_spatial_distance(self, bbox1: Tuple[int, int, int, int], 
                                  bbox2: Tuple[int, int, int, int], 
                                  frame_shape: Tuple[int, int] = (720, 1280)) -> float:
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
        
        # euclidian distance between centers
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        frame_h, frame_w = frame_shape
        normalized_distance = distance / (frame_w * 0.5) 
        
        return normalized_distance

    def calculate_combined_similarity(self, current_features: np.ndarray, 
                                    current_bbox: Tuple[int, int, int, int],
                                    player_info: Dict,
                                    frame_shape: Tuple[int, int] = (720, 1280)) -> float:
        if not player_info['bbox_history']:
            return 0.0
        
        last_bbox = player_info['bbox_history'][-1]
        spatial_distance = self.calculate_spatial_distance(current_bbox, last_bbox, frame_shape)
        
        if spatial_distance > self.max_distance_threshold:
            return 0.0
        
        spatial_similarity = max(0, 1 - spatial_distance * 2.0)
        
        visual_similarity = 0.5  
        if player_info['feature_history']:
            try:
                visual_similarities = []
                for hist_features in player_info['feature_history']:
                    sim = self.calculate_similarity(current_features, hist_features)
                    visual_similarities.append(sim)
                
                sorted_sims = sorted(visual_similarities, reverse=True)
                if len(sorted_sims) >= 2:
                    visual_similarity = (sorted_sims[0] * 0.6 + sorted_sims[1] * 0.4)
                else:
                    visual_similarity = sorted_sims[0]
            except:
                visual_similarity = 0.5 
        
        combined_similarity = (self.spatial_weight * spatial_similarity + 
                             self.feature_weight * visual_similarity)
        
        return combined_similarity
    
    def calculate_similarity(self, features1: np.ndarray, features2: np.ndarray) -> float:
        if len(features1) == 0 or len(features2) == 0:
            return 0.0
        
        features1 = features1.reshape(1, -1)
        features2 = features2.reshape(1, -1)
        
        similarity = cosine_similarity(features1, features2)[0, 0]
        return similarity

    def find_best_match(self, current_features: np.ndarray, 
                       current_bbox: Tuple[int, int, int, int]) -> Optional[int]:
        best_similarity = 0
        best_player_id = None
        
        candidates = []
        
        for player_id, player_info in self.players.items():
            if player_info['active']:
                continue 
            
            frames_missing = self.frame_count - player_info['last_seen']
            if frames_missing > self.max_missing_frames:
                continue
            
            combined_sim = self.calculate_combined_similarity(current_features, current_bbox, player_info)
            
            time_penalty = 1.0 - (frames_missing / self.max_missing_frames) * 0.3  # 30% penalty at max
            adjusted_similarity = combined_sim * time_penalty
            
            if adjusted_similarity > self.similarity_threshold:
                candidates.append((player_id, adjusted_similarity, frames_missing, combined_sim))
        
        candidates.sort(key=lambda x: (-x[1], x[2]))
        
        if candidates:
            best_player_id = candidates[0][0]
            best_similarity = candidates[0][1]
            
        return best_player_id
    
    def update_tracking(self, frame: np.ndarray, detections: List[Dict]):
        start_time = time.time()
        
        for player_info in self.players.values():
            player_info['active'] = False
        
        unassigned_detections = list(enumerate(detections))
        used_player_ids = set()
        frame_shape = frame.shape[:2]  
        
        for det_idx, detection in list(unassigned_detections):
            bbox = detection['bbox']
            features = self.feature_extractor.extract_features(frame, bbox)

            best_match_id = None
            best_similarity = 0
            
            candidates = []
            for player_id, player_info in self.players.items():
                if player_id in used_player_ids:
                    continue
                
                frames_missing = self.frame_count - player_info['last_seen']
                
                max_missing = self.max_missing_frames
                if frames_missing > max_missing:
                    continue
                
                combined_sim = self.calculate_combined_similarity(features, bbox, player_info, frame_shape)
                
                if frames_missing <= self.recent_frame_limit:
                    threshold = self.recent_threshold
                else:
                    threshold = self.old_threshold
                
                if combined_sim > threshold:
                    candidates.append((player_id, combined_sim, frames_missing))
            
            candidates.sort(key=lambda x: (-x[1], x[2]))
            
            if candidates:
                best_match_id = candidates[0][0]
                best_similarity = candidates[0][1]
                
                used_player_ids.add(best_match_id)
                detection['player_id'] = best_match_id
                
                player_info = self.players[best_match_id]
                
                if len(player_info['bbox_history']) > 0:
                    old_center = self.get_bbox_center(player_info['bbox_history'][-1])
                    new_center = self.get_bbox_center(bbox)
                    velocity = (new_center[0] - old_center[0], new_center[1] - old_center[1])
                    player_info['velocity'] = velocity
                
                player_info['feature_history'].append(features)
                player_info['bbox_history'].append(bbox)
                player_info['last_seen'] = self.frame_count
                player_info['active'] = True
                player_info['total_detections'] += 1
                player_info['class_id'] = detection.get('class_id', 2)
                
                unassigned_detections.remove((det_idx, detection))

        for det_idx, detection in list(unassigned_detections):
            bbox = detection['bbox']
            features = self.feature_extractor.extract_features(frame, bbox)
            
            reuse_candidates = []
            for player_id, player_info in self.players.items():
                if player_id not in used_player_ids and not player_info['active']:
                    if player_info['bbox_history']:
                        last_bbox = player_info['bbox_history'][-1]
                        spatial_dist = self.calculate_spatial_distance(bbox, last_bbox, frame_shape)
                        if spatial_dist < 0.6:  
                            reuse_candidates.append((player_id, spatial_dist))

            if reuse_candidates:
                reuse_candidates.sort(key=lambda x: x[1])  
                reuse_player_id = reuse_candidates[0][0]
                
                detection['player_id'] = reuse_player_id
                used_player_ids.add(reuse_player_id)

                player_info = self.players[reuse_player_id]
                player_info['feature_history'].append(features)
                player_info['bbox_history'].append(bbox)
                player_info['last_seen'] = self.frame_count
                player_info['active'] = True
                player_info['total_detections'] += 1
                player_info['class_id'] = detection.get('class_id', 2)
                
                unassigned_detections.remove((det_idx, detection))
        
        for det_idx, detection in unassigned_detections:
            bbox = detection['bbox']
            features = self.feature_extractor.extract_features(frame, bbox)
            
            assigned_id = self.next_player_id
            self.next_player_id += 1
            
            self.players[assigned_id] = {
                    'feature_history': deque(maxlen=self.feature_history_size),
                    'last_seen': self.frame_count,
                    'bbox_history': deque(maxlen=10),  
                    'active': True,
                    'total_detections': 1,
                    'class_id': detection.get('class_id', 2),
                    'velocity': None 
                }
            
            self.players[assigned_id]['feature_history'].append(features)
            self.players[assigned_id]['bbox_history'].append(bbox)
            detection['player_id'] = assigned_id
        
        tracking_time = time.time() - start_time
        self.tracking_times.append(tracking_time)
    
    def cleanup_old_players(self):
        cleanup_threshold = self.max_missing_frames * 3 
        
        players_to_remove = []
        for player_id, player_info in self.players.items():
            frames_missing = self.frame_count - player_info['last_seen']
            if frames_missing > cleanup_threshold and not player_info['active']:
                players_to_remove.append(player_id)
        
        for player_id in players_to_remove:
            del self.players[player_id]

    def process_frame(self, frame: np.ndarray) -> List[Dict]:
        detections = self.detect_players(frame)
        
        self.update_tracking(frame, detections)
        
        if self.frame_count % 30 == 0:  
            self.cleanup_old_players()

        self.frame_count += 1
        
        return detections
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        result_frame = frame.copy()
        
        player_colors = [
            (0, 255, 0),    # green
            (255, 0, 0),    # blue
            (0, 0, 255),    # red
            (255, 255, 0),  # cyan
            (255, 0, 255),  # magenta
            (0, 255, 255),  # yellow
            (128, 0, 128),  # purple
            (255, 165, 0),  # orange
            (0, 128, 255),  # light blue
            (128, 255, 0),  # lime
        ]
        
        goalkeeper_color = (0, 215, 255)  # gold/orange for goalkeeper
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            player_id = detection.get('player_id', -1)
            class_id = detection.get('class_id', 2)
            
            if player_id in self.players:
                stored_class_id = self.players[player_id].get('class_id', class_id)
                class_id = stored_class_id
            
            x1, y1, x2, y2 = map(int, bbox)
            
            if class_id == 1:  
                color = goalkeeper_color
                label_prefix = "GK"
            else:  
                color = player_colors[player_id % len(player_colors)] if player_id > 0 else (128, 128, 128)
                label_prefix = "P"
            
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 3)
            
            center_x = (x1 + x2) // 2
            id_y = y1 - 20
            if id_y < 35:
                id_y = y2 + 35
            
            circle_radius = 25 if class_id == 1 else 20
            cv2.circle(result_frame, (center_x, id_y), circle_radius, color, -1)
            cv2.circle(result_frame, (center_x, id_y), circle_radius, (255, 255, 255), 2)
            
            id_text = f"{label_prefix}{player_id}" if player_id > 0 else f"{label_prefix}?"
            font_scale = 0.7 if class_id == 1 else 0.6
            text_size = cv2.getTextSize(id_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = id_y + text_size[1] // 2
            cv2.putText(result_frame, id_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
            
            conf_text = f"{confidence:.2f}"
            conf_size = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            conf_x = center_x - conf_size[0] // 2
            conf_y = y2 + 20
            cv2.rectangle(result_frame, (conf_x - 5, conf_y - conf_size[1] - 5), 
                         (conf_x + conf_size[0] + 5, conf_y + 5), (0, 0, 0), -1)
            cv2.putText(result_frame, conf_text, (conf_x, conf_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        active_players = len([p for p in self.players.values() if p['active']])
        total_players = len(self.players)
        info_text = f"Frame: {self.frame_count} | Active: {active_players} | Total: {total_players}"
        
        text_size = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(result_frame, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
        cv2.putText(result_frame, info_text, (15, 15 + text_size[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return result_frame
    
    def get_performance_stats(self) -> Dict:
        active_players = [p for p in self.players.values() if p['active']]
        long_term_players = [p for p in self.players.values() if p['total_detections'] > 10]
        
        stats = {
            'total_frames': self.frame_count,
            'total_players': len(self.players),
            'active_players': len(active_players),
            'long_term_players': len(long_term_players),
            'avg_detection_time': np.mean(self.detection_times) if self.detection_times else 0,
            'avg_tracking_time': np.mean(self.tracking_times) if self.tracking_times else 0,
            'avg_fps': 1.0 / (np.mean(self.detection_times) + np.mean(self.tracking_times)) if (self.detection_times and self.tracking_times) else 0
        }
        
        return stats
    
    def get_bbox_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
