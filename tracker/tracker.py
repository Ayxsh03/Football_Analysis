from ultralytics import YOLO
import supervision as sv
import cv2
import pickle
import os
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from utils import get_center, get_width


class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_detections = self.model.predict(frames[i:i + batch_size], conf=0.1)
            detections += batch_detections
        return detections
    
    def draw_elipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3]) # bottom y-coordinate
        x_center, _ = get_center(bbox)
        width = get_width(bbox)

        cv2.ellipse(
            frame,
            (int(x_center), int(y2)),              # center
            (int(width), int(width / 4)),  # axes
            0,                           # angle
            -45,                          # startAngle
            235,                         # endAngle
            color,                       # color
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height = 20
        x1_rect = int(x_center - rectangle_width / 2)
        x2_rect = int(x_center + rectangle_width / 2)
        y1_rect = int(y2 - rectangle_height/2) + 15
        y2_rect = int(y2 + rectangle_height/2) + 15

        x1_text = x1_rect + 12
        if track_id is not None and track_id > 99:
            x1_text -= 10

        if(track_id is not None):
            cv2.rectangle(
                frame,
                (x1_rect, y1_rect),
                (x2_rect, y2_rect),
                color,
                cv2.FILLED
            )
            cv2.putText(
                frame,
                f"{track_id}",
                (x1_text, y1_rect + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2
            )
        return frame

    def draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x_center, _ = get_center(bbox)
        triangle_points = np.array([
            (int(x_center), int(y)),  # top point
            (int(x_center - 10), int(y - 20)),  # bottom left point
            (int(x_center + 10), int(y - 20))   # bottom right point
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, -1)
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)  # outline

        return frame

    def object_tracks(self, frames, read_from_stub=False, stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
                return tracks
        
        detections = self.detect_frames(frames)

        tracks={
            "player": [],
            "referee": [],
            "ball": []
            #"goalkeeper": []
        }

        for frame_num, detections in enumerate(detections):
            class_names = detections.names
            class_names_inverse = {v: k for k, v in class_names.items()}
            
            # Convert detections to supervision format
            detections_supervision = sv.Detections.from_ultralytics(detections)

            # Convert goalkeeper to player
            for object_id, class_id in enumerate(detections_supervision.class_id):
                if class_names[class_id] == 'goalkeeper':
                    detections_supervision.class_id[object_id] = class_names_inverse['player']


            # Track objects
            detection_tracks = self.tracker.update_with_detections(detections_supervision)

            tracks["player"].append({})
            tracks["referee"].append({})
            tracks["ball"].append({})
            #tracks["goalkeeper"].append({})

            for frame_detections in detection_tracks:
                bbox= frame_detections[0].tolist()
                class_id = frame_detections[3]
                track_id = frame_detections[4]

                if class_id == class_names_inverse['player']:
                    tracks["player"][frame_num][track_id] = {"bbox":bbox}
                if class_id == class_names_inverse['referee']:
                    tracks["referee"][frame_num][track_id] = {"bbox":bbox}
            
            for frame_detections in detections_supervision:
                bbox= frame_detections[0].tolist()
                class_id = frame_detections[3]
                if class_id == class_names_inverse['ball']:
                    tracks["ball"][frame_num][1] = {"bbox":bbox}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def draw_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_frame = team_ball_control[:frame_num + 1]
        team_1 = team_ball_control_frame[team_ball_control_frame==1].shape[0]
        team_2 = team_ball_control_frame[team_ball_control_frame==2].shape[0]

        total = team_1 + team_2
        team_1 = team_1 / total if total > 0 else 0
        team_2 = team_2 / total if total > 0 else 0
        cv2.putText(frame, f"Team 1: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, f"Team 2: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        annotated_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["player"][frame_num]
            ball_dict = tracks["ball"][frame_num]
            referee_dict = tracks["referee"][frame_num]
            
            for track_id, player in player_dict.items():
                colour = player.get("team_colours", (255, 0, 0))  # Default to blue if no colour is set
                frame = self.draw_elipse(frame, player["bbox"], colour, track_id)

                if player.get("has_ball", False):
                    frame = self.draw_triangle(frame, player["bbox"], (0, 0, 255))
            
            for _, referee in referee_dict.items():
                frame = self.draw_elipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():
                frame = self.draw_triangle(frame, ball["bbox"], (0, 255, 0))

            # Draw Ball Control
            frame = self.draw_ball_control(frame, frame_num, team_ball_control)
            annotated_frames.append(frame)

        return annotated_frames

