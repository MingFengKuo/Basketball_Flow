import torch
import torch.nn as nn

def dribbler_penalty(rec_x, real_x, status, basket_pos):
    # Get clostest distance between player and ball
    status = status.view(rec_x.size(0), rec_x.size(1), 6)
    real_penalty = dribbler_score(real_x, status, basket_pos).sum() / status.sum()
    fake_penalty = dribbler_score(rec_x, status, basket_pos).sum() / status.sum()
    return torch.abs(real_penalty - fake_penalty)

def dribbler_score(inputs, status, basket_pos):
    b_size, t_size, _ = inputs.size()
    # Basket pos
    basket_right_x = torch.FloatTensor(b_size, t_size, 1, 1).fill_(basket_pos[0]).cuda()
    basket_right_y = torch.FloatTensor(b_size, t_size, 1, 1).fill_(basket_pos[1]).cuda()
    basket_pos = torch.cat((basket_right_x, basket_right_y), dim=-1)
    # Team A players pos
    team_pos = inputs[:, :, 2:12].view(b_size, t_size, 5, 2)
    team_pos = torch.cat((team_pos, basket_pos), dim=-2)
    # Ball pos
    ball_pos = inputs[:, :, :2].view(b_size, t_size, 1, 2)
    # Compute
    dist = torch.norm((ball_pos - team_pos), p=2, dim=-1)
    dribbler_score = dist * status
    return dribbler_score

def blocked_penalty(rec_x, real_x, status, basket_pos):
    status = status.view(real_x.size(0), real_x.size(1), 6)
    real_penalty = blocked_score(real_x, status, basket_pos).sum() / status[:, :, :5].sum()
    fake_penalty = blocked_score(rec_x, status, basket_pos).sum() / status[:, :, :5].sum()
    return torch.abs(real_penalty - fake_penalty)
 
def blocked_score(inputs, status, basket_pos):
    b_size, t_size, _ = inputs.size()
    # Basket pos
    basket_right_x = torch.FloatTensor(b_size, t_size, 1, 1).fill_(basket_pos[0]).cuda()
    basket_right_y = torch.FloatTensor(b_size, t_size, 1, 1).fill_(basket_pos[1]).cuda()
    basket_pos = torch.cat((basket_right_x, basket_right_y), dim=-1)
    # Team B players pos
    team_pos = inputs[:, :, 12:22].view(b_size, t_size, 5, 2)
    # ball pos
    ball_pos = inputs[:, :, :2].view(b_size, t_size, 1, 2)
    
    vec_ball_2_team = ball_pos - team_pos
    vec_ball_2_basket = ball_pos - basket_pos
    b2team_dot_b2basket = torch.matmul(vec_ball_2_team,
                                        torch.transpose(vec_ball_2_basket, -2 ,-1))
    b2team_dot_b2basket = b2team_dot_b2basket.view(b_size, t_size, 5)
    dist_team = torch.norm(vec_ball_2_team, p=2, dim=-1)
    dist_basket = torch.norm(vec_ball_2_basket, p=2, dim=-1)
    theta = torch.acos(b2team_dot_b2basket / (dist_team * dist_basket + 1e-3))
    defender_score = (theta + 1.0) * (dist_team + 1.0)
    defender_values, defender_indices = torch.min(defender_score, dim=-1)
    dribble_frames = torch.eq(torch.sum(status[:, :, :5], dim=-1), 1)
    defender_score = defender_values * dribble_frames
    return defender_score

def ball_passing_penalty(inputs, status, mask):
    b_size, t_size, _ = inputs.size()
    # ball pos
    ball_pos = inputs[:, :, :2].view(b_size, t_size, 2)
    # Feature
    status = status + (1 - mask[:, :, :6])
    ballpass_frames = torch.eq(torch.sum(status, dim=-1), 0)[:, 1:-1]
    vel_1 = ball_pos[:, 1:-1] - ball_pos[:, 0:-2]
    vel_2 = ball_pos[:, 2:]   - ball_pos[:, 1:-1]
    dot_p = vel_1[:, :, 0] * vel_2[:, :, 0] + vel_1[:, :, 1] * vel_2[:, :, 1]
    vel_1_norm = torch.sqrt(vel_1[:, :, 0]**2 + vel_1[:, :, 1]**2 + 1e-10)
    vel_2_norm = torch.sqrt(vel_2[:, :, 0]**2 + vel_2[:, :, 1]**2 + 1e-10)
    v = dot_p / (vel_1_norm * vel_2_norm)
    clip = torch.clamp(v, -1.0+1e-5, 1.0-1e-5)
    theta = torch.acos(clip)
    pass_theta = ballpass_frames.float() * theta
    frames = torch.count_nonzero(ballpass_frames).float()
    ball_passing_score = pass_theta.sum()
    if frames > 0:
        ball_passing_score = ball_passing_score / frames
    return ball_passing_score

def velocity_penalty(rec_x, real_x, mask):
    real_penalty = velocity_score(real_x, mask)
    fake_penalty = velocity_score(rec_x, mask)
    return torch.abs(real_penalty - fake_penalty)

def velocity_score(inputs, mask):
    b_size, t_size, _ = inputs.size()
    vel = inputs[:, 1:, 2:22] - inputs[:, :-1, 2:22]
    vel = vel.view(b_size, t_size-1, 10, 2)
    dist = torch.norm(vel, p=2, dim=-1)
    mask = mask[:, 1:, 2:12]
    velocity_score = (dist * mask).sum() / mask.sum()
    return velocity_score
    
def acceleration_penalty(rec_x, real_x, mask):
    real_penalty = acceleration_score(real_x, mask)
    fake_penalty = acceleration_score(rec_x, mask)
    return torch.abs(real_penalty - fake_penalty)

def acceleration_score(inputs, mask):
    b_size, t_size, _ = inputs.size()
    vel = inputs[:, 1:, 2:22] - inputs[:, :-1, 2:22]
    acc = vel[:, 1:] - vel[:, :-1]
    acc = acc.view(b_size, t_size-2, 10, 2)
    dist = torch.norm(acc, p=2, dim=-1)
    mask = mask[:, 2:, 2:12]
    acceleration_score = (dist * mask).sum() / mask.sum()
    return acceleration_score

# WGAN-GP
def gradient_penalty(fake_play, real_play, critic):
    alpha = torch.rand((real_play.size(0), 1, 1)).cuda()
    interpolates = (alpha * real_play + (1 - alpha) * fake_play)
    interpolates = interpolates.requires_grad_(True)
    # Calculate probability of interpolated examples
    _, d_interpolates, _ = critic(interpolates)
    # Calculate gradients of probabilities with respect to examples
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones((real_play.size(0), 1)).cuda(),
                                    create_graph=True, retain_graph=True)[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)
    gradient_penalty = 10 * ((gradients_norm - 1) ** 2).mean()
    return gradient_penalty

def critic_penalty(fake_play, real_play, critic):
    r_feat, r_score, sketch = critic(real_play)
    f_feat, f_score, _ = critic(fake_play)
    grad_pen = gradient_penalty(fake_play, real_play, critic)
    # reconstruction loss
    r_loss = nn.MSELoss()(f_feat, r_feat)
    # discriminator loss
    d_loss = f_score.mean() - r_score.mean() + grad_pen
    # generator loss
    g_loss = -f_score.mean()
    return r_loss, g_loss, d_loss, sketch

