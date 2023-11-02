import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation

def update_all(frame_id, line_path, player_circles, ball_circle, annotations, data):
    # players
    for j, circle in enumerate(player_circles):
        x = 2 + j*2 + 0
        y = 2 + j*2 + 1
        circle.center = data[frame_id, x], data[frame_id, y]
        annotation_pos_x, annotation_pos_y = circle.center
        annotations[j].set_position((annotation_pos_x, annotation_pos_y-1.8))
    
    idx = 0
    count = 1
    for i, line in enumerate(line_path):
        idx = i % 11
        start = max((frame_id - count, 0))       
        line.set_data(data[start:frame_id+1, idx*2], data[start:frame_id+1, idx*2+1])
        if (i + 1) % 11 == 0:
            count += 1
                
    # ball
    ball_circle.center = data[frame_id, 0], data[frame_id, 1]

def video(data, length, path_length=24, file_path=None, fps=6, dpi=128):
    # basic data
    court = plt.imread("./data/court.png")  # 500*939
    name_list = ['A1', 'A2', 'A3', 'A4', 'A5', 'B1', 'B2', 'B3', 'B4', 'B5']
    
    # set players' circle
    player_circles = []
    [player_circles.append(plt.Circle(xy=(0, 0), radius=0.8, color='r')) for _ in range(5)]
    [player_circles.append(plt.Circle(xy=(0, 0), radius=0.8, color='b')) for _ in range(5)]
    
    # set ball's circle
    ball_circle = plt.Circle(xy=(0, 0), radius=0.6, color='g')
    
    # plot
    ax = plt.axes(xlim=(0, 100), ylim=(50, 0))
    ax.axis('off')
    
    # set line path
    line_path = []
    for x in range (path_length):
        alpha = 0.2 - (x * (0.2/path_length))
        linewidth = 3 - (x * (3/path_length))
        for i in range(11):
            if i == 0:
                line, = ax.plot([], [], alpha=alpha, linewidth=linewidth, c='g',
                                markersize=0, zorder=1, solid_capstyle='round')
            elif i < 6:
                line, = ax.plot([], [], alpha=alpha, linewidth=linewidth, c='r',
                                markersize=0, zorder=1, solid_capstyle='round',
                                marker='H', markeredgewidth=0)
            elif i >= 6:
                line, = ax.plot([], [], alpha=alpha, linewidth=linewidth, c='b',
                                markersize=0, zorder=1, solid_capstyle='round',
                                marker='H', markeredgewidth=0)
            line_path.append(line)
            
    # add players' circle
    for circle in player_circles:
        ax.add_patch(circle)
        
    # add ball's circle
    ax.add_patch(ball_circle)

    # annotations on circles
    annotations = [
        ax.annotate(name_list[i], xy=[0., 0.],
                    horizontalalignment='center', verticalalignment='center',
                    fontweight='bold', fontsize=6) for i in range(10)
    ]
    
    # animation
    fig = plt.gcf()
    anim = animation.FuncAnimation(fig,
                                   update_all,
                                   fargs=(line_path, player_circles, ball_circle, annotations, data),
                                   frames=length,
                                   interval=100)

    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    
    # save
    anim.save(file_path, fps=fps, dpi=dpi, writer='ffmpeg')

    # clear content
    plt.cla()
    plt.clf()
    
def trajectory(data, file_path):
    # team A -> red circle, team B -> red circle, ball -> small green circle
    court = plt.imread("data/court.png")  # 500*939
    data = data.transpose()
    
    # plot
    plt.axes(xlim=(0, 100), ylim=(50, 0))
    plt.axis('off')
    
    # --- offense ---
    # players (Team A)
    for i in range (5):
        plt.plot(data[2 + i*2 + 0], data[2 + i*2 + 1], '.-r', markersize=6)
        
    # ball
    plt.plot(data[0], data[1], '.-g', markersize=4)
    
    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    plt.savefig(file_path+'_offense.png')
    
    # clear content
    plt.cla()
    plt.clf()
    
    # plot
    plt.axes(xlim=(0, 100), ylim=(50, 0))
    plt.axis('off')
    
    # --- defense ---
    # players (Team B)
    for i in range (5,10):
        plt.plot(data[2 + i*2 + 0], data[2 + i*2 + 1], '.-b', markersize=6)
        
    plt.imshow(court, zorder=0, extent=[0, 100 - 6, 50, 0])
    plt.savefig(file_path+'_defense.png')
    
    # clear content
    plt.cla()
    plt.clf()
 
    
def visualize(data, length, path_length=24, save_path="./visualize"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    for i in range(len(data)):
        video(data[i, :length],
              length=length,
              path_length=path_length,
              file_path=save_path+'/video_'+str(i)+'.mp4')
        trajectory(data[i, :length],
                   file_path=save_path+'/trajectory_'+str(i))
    