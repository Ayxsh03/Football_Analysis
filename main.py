import numpy as np

from utils import read_video, save_video
from tracker import Tracker 
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_estimator import CameraEstimator


def main():
    video_frames = read_video('input/input.mp4')
    tracker = Tracker('models/best.pt')
    
    tracks = tracker.object_tracks(video_frames,read_from_stub=True, stub_path='stubs/tracks_stub.pkl')

    #interpolate
    tracks["ball"]=tracker.interpolate_ball_positions(tracks["ball"])

    team_assigner = TeamAssigner()
    team_assigner.assign_teams_color(video_frames[0], tracks['player'][0])
    
    for frame_num, player_track in enumerate(tracks['player']):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], track['bbox'], player_id)
            tracks['player'][frame_num][player_id]['team'] = team 
            tracks['player'][frame_num][player_id]['team_colours'] = team_assigner.team_colours[team]
    
    # camera movement estimator
    camera_movement_estimator = CameraEstimator(video_frames[0])
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                            read_from_stub=True,
                                                                            stub_path='stubs/camera_movement_stub.pkl')
    #camera_movement_estimator.add_adjust_positions_to_tracks(tracks,camera_movement_per_frame)

    
    # Assign ball to player
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks['player']):
        ball_bbox = tracks['ball'][frame_num][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks['ball'][frame_num][1]['assigned_player'] = assigned_player
            tracks['player'][frame_num][assigned_player]['has_ball'] = True
            team_ball_control.append(tracks['player'][frame_num][assigned_player]['team'])
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
    output_frames = camera_movement_estimator.draw_camera_movement(output_frames, camera_movement_per_frame)
    save_video(output_frames, 'runs/output.mp4')


if __name__ == "__main__":
    main()