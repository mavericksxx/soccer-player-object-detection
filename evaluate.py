# EVALUATION SCRIPT

# this script helps in evaluation of tracking performance including
# metrics for reidentification accuracy, consistency, and timing.
# script starts with initializing the evaluator class with the results we obtained.
# it then builds builds player trajectories (per frame)
# then we calculate other metrics: trajectory consistency, detection stability, and id consistency
# and finally we print a report and generate plots for visualization

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import argparse
from collections import defaultdict


class TrackingEvaluator:
    
    def __init__(self, results_path: str):
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.player_trajectories = self._build_trajectories()
        self.metrics = {}
    
    def _build_trajectories(self) -> Dict[int, List[Dict]]:
        trajectories = defaultdict(list)
        
        for frame_data in self.results:
            frame_num = frame_data['frame']
            timestamp = frame_data['timestamp']
            
            for detection in frame_data['detections']:
                player_id = detection['player_id']
                
                trajectory_point = {
                    'frame': frame_num,
                    'timestamp': timestamp,
                    'bbox': detection['bbox'],
                    'confidence': detection['confidence']
                }
                
                trajectories[player_id].append(trajectory_point)
        
        return dict(trajectories)
    
    def calculate_trajectory_consistency(self) -> Dict[str, float]:
        consistency_scores = []
        temporal_gaps = []
        
        for player_id, trajectory in self.player_trajectories.items():
            if len(trajectory) < 2:
                continue
            
            trajectory = sorted(trajectory, key=lambda x: x['frame'])
            
            frame_numbers = [point['frame'] for point in trajectory]
            gaps = np.diff(frame_numbers)
            
            re_id_events = np.sum(gaps > 1)
            max_gap = np.max(gaps) if len(gaps) > 0 else 0
            
            temporal_gaps.extend(gaps)
            
            centers = []
            for point in trajectory:
                bbox = point['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                centers.append([center_x, center_y])
            
            if len(centers) > 2:
                centers = np.array(centers)
                velocities = np.diff(centers, axis=0)
                accelerations = np.diff(velocities, axis=0)
                smoothness = 1.0 / (1.0 + np.mean(np.linalg.norm(accelerations, axis=1)))
                consistency_scores.append(smoothness)
        
        return {
            'avg_consistency': np.mean(consistency_scores) if consistency_scores else 0,
            'avg_temporal_gap': np.mean(temporal_gaps) if temporal_gaps else 0,
            'max_temporal_gap': np.max(temporal_gaps) if temporal_gaps else 0,
            'total_re_id_events': np.sum(np.array(temporal_gaps) > 1)
        }
    
    def calculate_detection_stability(self) -> Dict[str, float]:
        frame_counts = []
        confidence_scores = []
        bbox_stability = []
        
        for player_id, trajectory in self.player_trajectories.items():
            frame_counts.append(len(trajectory))
            
            confidences = [point['confidence'] for point in trajectory]
            confidence_scores.extend(confidences)
            
            if len(trajectory) > 1:
                sizes = []
                for point in trajectory:
                    bbox = point['bbox']
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    sizes.append(width * height)
                
                if len(sizes) > 1:
                    size_std = np.std(sizes) / np.mean(sizes)  # Coefficient of variation
                    bbox_stability.append(1.0 / (1.0 + size_std))
        
        total_frames = len(self.results)
        
        return {
            'avg_detection_per_player': np.mean(frame_counts) if frame_counts else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'confidence_std': np.std(confidence_scores) if confidence_scores else 0,
            'avg_bbox_stability': np.mean(bbox_stability) if bbox_stability else 0,
            'frame_coverage': np.mean(frame_counts) / total_frames if frame_counts and total_frames > 0 else 0
        }
    
    def calculate_id_consistency(self) -> Dict[str, float]:
        position_history = defaultdict(list)
        
        for frame_data in self.results:
            for detection in frame_data['detections']:
                player_id = detection['player_id']
                bbox = detection['bbox']
                center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                
                position_history[frame_data['frame']].append({
                    'id': player_id,
                    'center': center,
                    'bbox': bbox
                })
        
        id_switches = 0
        frames = sorted(position_history.keys())
        
        for i in range(1, len(frames)):
            current_frame = frames[i]
            prev_frame = frames[i-1]
            
            current_detections = position_history[current_frame]
            prev_detections = position_history[prev_frame]
            
            for curr_det in current_detections:
                for prev_det in prev_detections:
                    dist = np.sqrt((curr_det['center'][0] - prev_det['center'][0])**2 + 
                                 (curr_det['center'][1] - prev_det['center'][1])**2)
                    # if the dist is small enough and ids happen to be different, we count it as an id switch
                    if dist < 50 and curr_det['id'] != prev_det['id']:
                        id_switches += 1
        
        unique_ids = len(set(det['player_id'] for frame in self.results for det in frame['detections']))
        
        return {
            'total_unique_ids': unique_ids,
            'potential_id_switches': id_switches,
            'id_consistency_score': 1.0 - (id_switches / max(1, len(frames) * unique_ids))
        }
    
    def evaluate_all(self) -> Dict:
        print("Evaluating tracking performance...")
        
        trajectory_metrics = self.calculate_trajectory_consistency()
        stability_metrics = self.calculate_detection_stability()
        id_metrics = self.calculate_id_consistency()
        
        all_metrics = {
            'trajectory': trajectory_metrics,
            'stability': stability_metrics,
            'identity': id_metrics
        }
        
        consistency_score = trajectory_metrics['avg_consistency']
        stability_score = stability_metrics['avg_bbox_stability']
        id_score = id_metrics['id_consistency_score']
        
        overall_score = (consistency_score + stability_score + id_score) / 3
        all_metrics['overall_score'] = overall_score
        
        self.metrics = all_metrics
        return all_metrics
    
    def print_report(self):
        if not self.metrics:
            self.evaluate_all()
        
        print("\n" + "="*60)
        print("SOCCER PLAYER TRACKING EVALUATION REPORT")
        print("="*60)
        
        print(f"\n OVERALL SCORE: {self.metrics['overall_score']:.3f}")
        
        print("\n TRAJECTORY CONSISTENCY:")
        traj = self.metrics['trajectory']
        print(f"  • Average Consistency: {traj['avg_consistency']:.3f}")
        print(f"  • Re-identification Events: {traj['total_re_id_events']}")
        print(f"  • Max Gap (frames): {traj['max_temporal_gap']}")
        print(f"  • Average Gap: {traj['avg_temporal_gap']:.1f} frames")
        
        print("\n DETECTION STABILITY:")
        stab = self.metrics['stability']
        print(f"  • Average Detections per Player: {stab['avg_detection_per_player']:.1f}")
        print(f"  • Frame Coverage: {stab['frame_coverage']:.1%}")
        print(f"  • Average Confidence: {stab['avg_confidence']:.3f}")
        print(f"  • Bbox Stability: {stab['avg_bbox_stability']:.3f}")
        
        print("\n IDENTITY MANAGEMENT:")
        ident = self.metrics['identity']
        print(f"  • Total Unique IDs: {ident['total_unique_ids']}")
        print(f"  • Potential ID Switches: {ident['potential_id_switches']}")
        print(f"  • ID Consistency Score: {ident['id_consistency_score']:.3f}")
        
        print(f"\n PLAYER SUMMARY:")
        print(f"  • Total Players Tracked: {len(self.player_trajectories)}")
        
        for player_id, trajectory in self.player_trajectories.items():
            duration = len(trajectory)
            gaps = []
            
            frames = sorted([point['frame'] for point in trajectory])
            if len(frames) > 1:
                gaps = np.diff(frames)
                re_entries = np.sum(gaps > 1)
            else:
                re_entries = 0
            
            print(f"    Player {player_id}: {duration} detections, {re_entries} re-entries")
    
    def plot_trajectories(self, save_path: str = 'trajectories.png'):
        plt.figure(figsize=(12, 8))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.player_trajectories)))
        
        for i, (player_id, trajectory) in enumerate(self.player_trajectories.items()):
            if len(trajectory) < 2:
                continue
            
            x_coords = []
            y_coords = []
            
            for point in sorted(trajectory, key=lambda x: x['frame']):
                bbox = point['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                x_coords.append(center_x)
                y_coords.append(center_y)
            
            plt.plot(x_coords, y_coords, 'o-', color=colors[i], 
                    label=f'Player {player_id}', alpha=0.7, markersize=3)
        
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('Player Trajectories')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trajectory plot saved to: {save_path}")
    
    def plot_timeline(self, save_path: str = 'timeline.png'):
        plt.figure(figsize=(15, 8))
        
        player_ids = sorted(self.player_trajectories.keys())
        
        for i, player_id in enumerate(player_ids):
            trajectory = self.player_trajectories[player_id]
            frames = [point['frame'] for point in trajectory]
            y_pos = [i] * len(frames)
            
            plt.scatter(frames, y_pos, alpha=0.7, s=20, label=f'Player {player_id}')
        
        plt.xlabel('Frame Number')
        plt.ylabel('Player ID')
        plt.title('Player Detection Timeline')
        plt.yticks(range(len(player_ids)), [f'Player {pid}' for pid in player_ids])
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Timeline plot saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Soccer Player Tracking Results')
    parser.add_argument('--results', '-r', default='tracking_results.json',
                       help='Path to tracking results JSON file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    
    args = parser.parse_args()
    
    try:
        evaluator = TrackingEvaluator(args.results)
        
        metrics = evaluator.evaluate_all()
        
        evaluator.print_report()
        
        if args.plot:
            evaluator.plot_trajectories()
            evaluator.plot_timeline()
        
        metrics_path = 'evaluation_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n Detailed metrics saved to: {metrics_path}")
        
    except FileNotFoundError:
        print(f"Error: Results file not found: {args.results}")
        print("Run the tracking first to generate results.")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
