import torch

from torch import nn
from torch import optim

from math import log, pi
from models.networks import Extractor, Encoder, Generator, Discriminator
from models.loss import (critic_penalty, dribbler_penalty, ball_passing_penalty, blocked_penalty,
                         acceleration_penalty, velocity_penalty)

from models.flow import get_cnf

class Basketball_Flow(nn.Module):
    def __init__(self, args):
        super(Basketball_Flow, self).__init__()
        # basic
        self.gpu   = args.gpu
        self.f_dim = args.f_dim
        self.c_dim = args.c_dim
        self.t_dim = args.t_dim
        self.z_dim = args.z_dim
        
        # weight
        self.kl_weight = args.kl_weight
        self.recon_weight = args.recon_weight
        self.sample_weight = args.sample_weight
        self.feature_weight = args.feature_weight
        self.relation_weight = args.relation_weight
        self.speed_weight = args.speed_weight
        
        """ Model """
        # Encoder
        self.encoder = \
            Encoder(z_dim=self.z_dim, f_dim=self.f_dim + self.c_dim, t_dim=self.t_dim,
                    patch_num=args.encode_patch_num, depth=args.encode_depth)
            
        # Generator
        self.generator = \
            Generator(z_dim=self.z_dim, f_dim=self.f_dim, t_dim=self.t_dim,
                      patch_num=args.generate_patch_num, depth=args.generate_depth)
        
        # Discriminator
        self.discriminator = \
            Discriminator(z_dim=self.z_dim, c_dim=self.c_dim, f_dim=self.f_dim, t_dim=self.t_dim,
                          depth=args.critic_depth)
            
        # Extractor
        self.extractor = \
            Extractor(z_dim=self.z_dim, f_dim=self.c_dim, t_dim=self.t_dim,
                     patch_num=args.extract_patch_num, depth=args.extract_depth)
            
        # Conditional flow
        self.ccnf = \
            get_cnf(args=args, input_dim=self.z_dim, hidden_dims=str(args.latent_hidden_dim),
                    context_dim=self.z_dim, num_blocks=args.latent_num_blocks,
                    conditional=True)
            
        """ optimizer """
        # Discriminator
        self.discriminator_optimizer = self.make_optimizer(list(self.discriminator.parameters()), args.d_lr, args)
        # Generator
        self.generator_optimizer = self.make_optimizer(list(self.encoder.parameters()) +
                                                       list(self.generator.parameters()), args.g_lr, args)
        # Flow
        self.ccnf_optimizer = self.make_optimizer(list(self.extractor.parameters()) + 
                                                  list(self.ccnf.parameters()), args.g_lr * 0.1, args)
        
    def make_optimizer(self, params, lr, args):
        if args.optimizer == 'sgd':
            optimizer = optim.SGD(params, lr=lr, momentum=args.momentum)
        elif args.optimizer == 'adam':
            optimizer = optim.Adam(params, lr=lr, betas=(args.beta1, args.beta2))
        elif args.optimizer == 'rmsprop':
            optimizer = optim.RMSprop(params, lr=lr, momentum=args.momentum)
        else:
            assert 0, "args.optimizer should be 'sgd', 'adam' or 'rmsprop'"
        return optimizer
    
    @staticmethod
    def reparameterize_gaussian(mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)
    
    @staticmethod
    def kl_divergence(mu, log_var):
        return torch.mean(-0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1))
        
    def train_D(self, x, c, mask):
        real_x = x
        cond_c = c
        
        """ Generate """
        # encode
        latent_z_mu, latent_z_var = self.encoder(torch.cat((cond_c, real_x), dim=-1))
        latent_z = self.reparameterize_gaussian(latent_z_mu, latent_z_var)
        # generate
        fake_x = self.generator(latent_z) * mask[:, :, :1]
        
        """ Train Critic / Discriminator"""
        self.discriminator_optimizer.zero_grad()
        # compute critic loss
        _, _, critic_loss, sketch = critic_penalty(fake_x.detach(), real_x, self.discriminator)
        # compute sketch reconstruct loss
        sketch = sketch * mask[:, :, :1]
        sket_recon_loss = nn.MSELoss(reduction='sum')(sketch[:, :, :12], cond_c[:, :, :12]) / mask[:, :, :12].sum()
        stat_recon_loss = nn.BCELoss(reduction='sum')(sketch[:, :, 12:], cond_c[:, :, 12:]) / mask[:, :, :6].sum()
        # update
        loss = critic_loss + sket_recon_loss + stat_recon_loss
        loss.backward()
        self.discriminator_optimizer.step()
        
    def train_G(self, x, c, mask, norm_dict, step, writer=None):
        # data
        real_x = x
        cond_c = c
        move_x = x[:, :, :22]
        stat_x = x[:, :, 22:]
        # basket position
        basket_pos = [norm_dict['x']['basket'], norm_dict['y']['basket']]
        
        """ Generate """
        # encode
        latent_z_mu, latent_z_var = self.encoder(torch.cat((cond_c, real_x), dim=-1))
        latent_z = self.reparameterize_gaussian(latent_z_mu, latent_z_var)
        # generate
        fake_x = self.generator(latent_z) * mask[:, :, :1]
        
        """ Train Generator """
        self.generator_optimizer.zero_grad()
        
        # compute kl-divergence
        kl_loss = self.kl_divergence(latent_z_mu, latent_z_var)
        
        # compute generate loss
        recon_loss, sample_loss, critic_loss, sketch = critic_penalty(fake_x, real_x, self.discriminator)
        self.discriminator.zero_grad() # prevent memory leaks

        # compute feature loss
        dribbler_loss     = dribbler_penalty(fake_x, move_x, stat_x, basket_pos)
        blocked_loss      = blocked_penalty(fake_x, move_x, stat_x, basket_pos)
        velocity_loss     = velocity_penalty(fake_x, move_x, mask)
        acceleration_loss = acceleration_penalty(fake_x, move_x, mask)
        ball_pass_loss    = ball_passing_penalty(fake_x, stat_x, mask)
        
        feature_loss = ball_pass_loss + (dribbler_loss + blocked_loss) * self.relation_weight
        feature_loss = feature_loss + (velocity_loss + acceleration_loss) * self.speed_weight
        
        # compute sketch reconstruct loss
        sketch = sketch * mask[:, :, :1]
        sket_recon_loss = nn.MSELoss(reduction='sum')(sketch[:, :, :12], cond_c[:, :, :12]) / mask[:, :, :12].sum()
        stat_recon_loss = nn.BCELoss(reduction='sum')(sketch[:, :, 12:], cond_c[:, :, 12:]) / mask[:, :, :6].sum()
        
        # update
        loss = (kl_loss * self.kl_weight +
                recon_loss * self.recon_weight +
                sample_loss * self.sample_weight + 
                feature_loss * self.feature_weight)
        loss.backward()
        self.generator_optimizer.step()
        
        """ Record """
        if writer is not None:
            writer.add_scalar('train-gan/kl-div',           kl_loss, step)
            writer.add_scalar('train-gan/recon',            recon_loss, step)
            writer.add_scalar('train-gan/critic',           critic_loss, step)
            writer.add_scalar('train-gan/sample',           sample_loss, step)
            writer.add_scalar('train-feature/dribbler',     dribbler_loss, step)
            writer.add_scalar('train-feature/blocked',      blocked_loss, step)
            writer.add_scalar('train-feature/ball_pass',    ball_pass_loss, step)
            writer.add_scalar('train-feature/velocity',     velocity_loss, step)
            writer.add_scalar('train-feature/acceleration', acceleration_loss, step)
            writer.add_scalar('train-sketch-recon/sketch',  sket_recon_loss, step)
            writer.add_scalar('train-sketch-recon/status',  stat_recon_loss, step)
        
        """ Return """
        return {
            'kl-div':        kl_loss.detach(),
            'recon':         recon_loss.detach(),
            'critic':        critic_loss.detach(),
            'sample':        sample_loss.detach(),
            'dribbler':      dribbler_loss.detach(),
            'blocked':       blocked_loss.detach(),
            'ball_pass':     ball_pass_loss.detach(),
            'velocity':      velocity_loss.detach(),
            'acceleration':  acceleration_loss.detach(),
            'sketch_recon':  sket_recon_loss.detach(),
            'status_recon':  stat_recon_loss.detach(),
        }
    
    def train_F(self, x, c, mask, norm_dict, step, writer=None):
        # data
        real_x = x
        cond_c = c
        
        """ Flow """
        # encode
        latent_z_mu, latent_z_var = self.encoder(torch.cat((cond_c, real_x), dim=-1))
        latent_z = self.reparameterize_gaussian(latent_z_mu, latent_z_var)
        # extract condtion
        ccnf_c = self.extractor(c)
        # condtional flow
        ccnf_w, delta_log_pw = self.ccnf(latent_z.detach(), ccnf_c, torch.zeros(latent_z.size(0), 1).to(latent_z))
        
        """ Train Flow """
        self.ccnf_optimizer.zero_grad()
        def logprob(z):
            dim = z.size(-1)
            log_z = -0.5 * dim * log(2 * pi)
            return log_z - z.pow(2).sum(dim=-1) / 2

        log_pw = logprob(ccnf_w).view(latent_z.size(0), -1)
        log_pz = log_pw - delta_log_pw
        prior_loss = -log_pz.mean()
            
        # update
        prior_loss.backward()
        self.ccnf_optimizer.step()
        
        """ Record """
        if writer is not None:
            writer.add_scalar('train-flow/prior', prior_loss, step)
            
        """ Return """
        return {
            'prior': prior_loss.detach(),
        }

    def validate(self, batch_size, x, c, mask, norm_dict):
        # data
        real_x = x
        cond_c = c
        move_x = x[:, :, :22]
        stat_x = x[:, :, 22:]
        # basket position
        basket_pos = [norm_dict['x']['basket'], norm_dict['y']['basket']]
        
        """ Generate """
        # encode
        latent_z_mu, latent_z_var = self.encoder(torch.cat((cond_c, real_x), dim=-1))
        latent_z = self.reparameterize_gaussian(latent_z_mu, latent_z_var)
        # generate
        fake_x = self.generator(latent_z) * mask[:, :, :1]
        
        """ Flow """
        ccnf_c = self.extractor(c)
        ccnf_w, delta_log_pw = self.ccnf(latent_z, ccnf_c, torch.zeros(latent_z.size(0), 1).to(latent_z))
       
        """ Sample """
        # sample
        latent_w = torch.randn(latent_z.size(0), self.z_dim).float().cuda()
        # reverse
        latent_z = self.ccnf(latent_w, ccnf_c, reverse=True)
        # generate
        sample_x = self.generator(latent_z) * mask[:, :, :1]
        
        """ Validate """
        # compute kl-divergence
        kl_loss = self.kl_divergence(latent_z_mu, latent_z_var)
        
        # compute generate loss
        r_feat, r_score, sketch = self.discriminator(real_x)
        f_feat, f_score, _ = self.discriminator(fake_x)
        recon_loss = nn.MSELoss()(f_feat, r_feat)
        em_dist = r_score.mean() - f_score.mean()
        
        # compute feature loss
        dribbler_loss     = dribbler_penalty(fake_x[:,:,:22], move_x, stat_x, basket_pos)
        blocked_loss      = blocked_penalty(fake_x[:,:,:22], move_x, stat_x, basket_pos)
        velocity_loss     = velocity_penalty(fake_x[:,:,:22], move_x, mask)
        acceleration_loss = acceleration_penalty(fake_x[:,:,:22], move_x, mask)
        ball_pass_loss    = ball_passing_penalty(fake_x[:,:,:22], stat_x, mask)
        
        # compute sketch reconstruct loss
        sketch = sketch * mask[:, :, :1]
        sket_recon_loss = nn.MSELoss(reduction='sum')(sketch[:, :, :12], cond_c[:, :, :12]) / mask[:, :, :12].sum()
        stat_recon_loss = nn.BCELoss(reduction='sum')(sketch[:, :, 12:], cond_c[:, :, 12:]) / mask[:, :, :6].sum()
        
        # compute prior loss
        def logprob(z):
            dim = z.size(-1)
            log_z = -0.5 * dim * log(2 * pi)
            return log_z - z.pow(2).sum(dim=-1) / 2

        log_pw = logprob(ccnf_w).view(latent_z.size(0), -1)
        log_pz = log_pw - delta_log_pw
        prior_loss = -log_pz.mean()
        
        return {
            'fake_x':       fake_x[:, :, :22],
            'sample_x':     sample_x[:, :, :22],
            'kl-div':       kl_loss,
            'recon':        recon_loss,
            'em-dist':      em_dist,
            'dribbler':     dribbler_loss,
            'blocked':      blocked_loss,
            'ball_pass':    ball_pass_loss,
            'velocity':     velocity_loss,
            'acceleration': acceleration_loss,
            'sketch_recon': sket_recon_loss,
            'status_recon': stat_recon_loss,
            'prior':        prior_loss,
         }
    
    def test_recon(self, x, c, mask, norm_dict, samples=4):
        real_x = x.repeat(samples, 1, 1)
        cond_c = c.repeat(samples, 1, 1)
        # encode
        latent_z_mu, latent_z_var = self.encoder(torch.cat((cond_c, real_x), dim=-1))
        latent_z = self.reparameterize_gaussian(latent_z_mu, latent_z_var)
        # generate
        fake_x = self.generator(latent_z) * mask[:, :, :1]
        return fake_x[:, :, :22]

    def test_sample(self, x, c, mask, norm_dict, samples=4):
        cond_c = c.repeat(samples, 1, 1)
        # sample
        ccnf_w = torch.randn(samples, self.z_dim).float().cuda()
        # reverse
        ccnf_c = self.extractor(cond_c)
        ccnf_z = self.ccnf(ccnf_w, ccnf_c, reverse=True)
        # generate
        sample_x = self.generator(ccnf_z)
        return sample_x[:, :, :22]