# MAIN APP

# this script processes the input video (15sec_input_720p.mp4) and demo's real time player tracking
# with reidentification
# the process_video function handles the video processing, player detection, and tracking 
# with the following args:
# - input_path: input video path
# - model_path: YOLO model path
# - output_path: output video path (optional)
# - display: option to display results in real time
# - save_results: option to save tracking results


import cv2
import argparse
import os
import json
from player_tracker import PlayerTracker
import time


def process_video(input_path: str, model_path: str, output_path: str = None, 
                 display: bool = True, save_results: bool = True):
    tracker = PlayerTracker(model_path)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    all_results = []
    frame_idx = 0
    
    target_frame_time = 1.0 / fps
    
    try:
        while True:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break

            detections = tracker.process_frame(frame)

            result_frame = tracker.draw_results(frame, detections)

            frame_results = {
                'frame': frame_idx,
                'timestamp': frame_idx / fps,
                'detections': []
            }
            # convert numpy types to py types for json serialization
            for detection in detections:
                bbox = detection['bbox']
                if hasattr(bbox, 'tolist'):
                    bbox = bbox.tolist()
                else:
                    bbox = [float(x) for x in bbox]
                
                frame_results['detections'].append({
                    'player_id': detection.get('player_id', -1),
                    'bbox': bbox,
                    'confidence': float(detection['confidence']),
                    'class_id': int(detection.get('class_id', 2))
                })
            
            all_results.append(frame_results)

            if display:
                cv2.imshow('Soccer Player Tracking', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if out:
                out.write(result_frame)
            
            processing_time = time.time() - start_time
            if processing_time < target_frame_time:
                time.sleep(target_frame_time - processing_time)
            
            frame_idx += 1
            
            if frame_idx % 30 == 0: 
                progress = (frame_idx / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_idx}/{total_frames}")
    
    except KeyboardInterrupt:
        print("Processing interrupted by user")
    
    finally:
        cap.release()
        if out:
            out.release()
        if display:
            cv2.destroyAllWindows()
    
    if save_results:
        results_path = 'tracking_results.json'
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Results saved to: {results_path}")
    
    stats = tracker.get_performance_stats()
    print("\n--- Performance Statistics ---")
    print(f"Total frames processed: {stats['total_frames']}")
    print(f"Total unique players detected: {stats['total_players']}")
    print(f"Average detection time: {stats['avg_detection_time']:.4f}s")
    print(f"Average tracking time: {stats['avg_tracking_time']:.4f}s")
    print(f"Average FPS: {stats['avg_fps']:.2f}")
    
    print("\n--- Player Statistics ---")
    active_players = sum(1 for p in tracker.players.values() if p['active'])
    total_detections = sum(p['total_detections'] for p in tracker.players.values())
    print(f"Currently active players: {active_players}")
    print(f"Total detections: {total_detections}")
    
    for player_id, info in tracker.players.items():
        status = "Active" if info['active'] else "Inactive"
        print(f"Player {player_id}: {info['total_detections']} detections, {status}")
    
    return all_results, stats


def analyze_results(results_path: str):
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    print("\n=== Analysis Results ===")
    
    player_appearances = {}
    player_frame_ranges = {}
    
    for frame_data in results:
        frame_num = frame_data['frame']
        for detection in frame_data['detections']:
            player_id = detection['player_id']
            
            if player_id not in player_appearances:
                player_appearances[player_id] = 0
                player_frame_ranges[player_id] = {'first': frame_num, 'last': frame_num}
            
            player_appearances[player_id] += 1
            player_frame_ranges[player_id]['last'] = frame_num
    
    re_id_events = []
    for player_id, ranges in player_frame_ranges.items():
        frames_with_player = []
        for frame_data in results:
            if any(d['player_id'] == player_id for d in frame_data['detections']):
                frames_with_player.append(frame_data['frame'])
        
        if len(frames_with_player) > 1:
            gaps = []
            for i in range(1, len(frames_with_player)):
                gap = frames_with_player[i] - frames_with_player[i-1]
                if gap > 1: 
                    gaps.append(gap)
            
            if gaps:
                re_id_events.append({
                    'player_id': player_id,
                    'gaps': gaps,
                    'max_gap': max(gaps),
                    'total_appearances': len(frames_with_player)
                })
    
    print(f"Players detected: {len(player_appearances)}")
    for player_id, count in player_appearances.items():
        print(f"Player {player_id}: {count} detections")
    
    print(f"\nRe-identification events: {len(re_id_events)}")
    for event in re_id_events:
        print(f"Player {event['player_id']}: Max gap {event['max_gap']} frames, "
              f"{len(event['gaps'])} re-entries")


def main():
    parser = argparse.ArgumentParser(description='Soccer Player Detection and Tracking')
    parser.add_argument('--input', '-i', default='15sec_input_720p.mp4',
                       help='Input video path')
    parser.add_argument('--model', '-m', default='best.pt',
                       help='YOLO model path')
    parser.add_argument('--output', '-o', default='output_tracked.mp4',
                       help='Output video path')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable real-time display')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze existing results')
    
    args = parser.parse_args()
    
    if args.analyze:
        if os.path.exists('tracking_results.json'):
            analyze_results('tracking_results.json')
        else:
            print("No results file found. Run tracking first.")
        return
    
    if not os.path.exists(args.input):
        print(f"Error: Input video not found: {args.input}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    print("Starting soccer player detection and tracking...")
    print(f"Input: {args.input}")
    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    
    results, stats = process_video(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        display=not args.no_display
    )
    
    print("\nProcessing completed successfully.")
    print("Run with --analyze flag to analyze results.")


if __name__ == "__main__":
    main()
