import torch
import numpy as np
import abc
from tqdm import trange

from losses import get_score_fn, get_score_fn_dpm_compt
from utils.graph_utils import mask_adjs, mask_x, gen_noise
from sde import VPSDE, subVPSDE


class Predictor(abc.ABC):
  """The abstract class for a predictor algorithm."""
  def __init__(self, sde, score_fn, probability_flow=False):
    super().__init__()
    self.sde = sde
    # Compute the reverse SDE/ODE
    self.rsde = sde.reverse(score_fn, probability_flow)
    self.score_fn = score_fn

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class Corrector(abc.ABC):
  """The abstract class for a corrector algorithm."""
  def __init__(self, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__()
    self.sde = sde
    self.score_fn = score_fn
    self.snr = snr
    self.scale_eps = scale_eps
    self.n_steps = n_steps

  @abc.abstractmethod
  def update_fn(self, x, t, flags):
    pass


class EulerMaruyamaPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    dt = -1. / self.rsde.N

    if self.obj=='x':
      z = gen_noise(x, flags, sym=False)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=False)
      x_mean = x + drift * dt
      x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
      return x, x_mean

    elif self.obj=='adj':
      z = gen_noise(adj, flags)
      drift, diffusion = self.rsde.sde(x, adj, flags, t, is_adj=True)
      adj_mean = adj + drift * dt
      adj = adj_mean + diffusion[:, None, None] * np.sqrt(-dt) * z

      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class ReverseDiffusionPredictor(Predictor):
  def __init__(self, obj, sde, score_fn, probability_flow=False):
    super().__init__(sde, score_fn, probability_flow)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):

    if self.obj == 'x':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=False)
      z = gen_noise(x, flags, sym=False)
      x_mean = x - f
      x = x_mean + G[:, None, None] * z
      return x, x_mean

    elif self.obj == 'adj':
      f, G = self.rsde.discretize(x, adj, flags, t, is_adj=True)
      z = gen_noise(adj, flags)
      adj_mean = adj - f
      adj = adj_mean + G[:, None, None] * z
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class NoneCorrector(Corrector):
  """An empty corrector that does nothing."""

  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    self.obj = obj
    pass

  def update_fn(self, x, adj, flags, t):
    if self.obj == 'x':
      return x, x
    elif self.obj == 'adj':
      return adj, adj
    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported.")


class LangevinCorrector(Corrector):
  def __init__(self, obj, sde, score_fn, snr, scale_eps, n_steps):
    super().__init__(sde, score_fn, snr, scale_eps, n_steps)
    self.obj = obj

  def update_fn(self, x, adj, flags, t):
    sde = self.sde
    score_fn = self.score_fn
    n_steps = self.n_steps
    target_snr = self.snr
    seps = self.scale_eps

    if isinstance(sde, VPSDE) or isinstance(sde, subVPSDE):
      timestep = (t * (sde.N - 1) / sde.T).long()
      alpha = sde.alphas.to(t.device)[timestep]
    else:
      alpha = torch.ones_like(t)

    if self.obj == 'x':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * grad
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return x, x_mean

    elif self.obj == 'adj':
      for i in range(n_steps):
        grad = score_fn(x, adj, flags, t)
        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * grad
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * seps
      return adj, adj_mean

    else:
      raise NotImplementedError(f"obj {self.obj} not yet supported")


# -------- PC sampler --------
def get_pc_sampler(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None',
                   snr=0.1, scale_eps=1.0, n_steps=1,
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):

  def pc_sampler(model_x, model_adj, init_flags, diff_steps=None):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    predictor_fn = ReverseDiffusionPredictor if predictor=='Reverse' else EulerMaruyamaPredictor
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector

    predictor_obj_x = predictor_fn('x', sde_x, score_fn_x, probability_flow)
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)

    predictor_obj_adj = predictor_fn('adj', sde_adj, score_fn_adj, probability_flow)
    corrector_obj_adj = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      if diff_steps is None:
        diff_steps = sde_adj.N
      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)

      print("Total number of sampling steps: ", diff_steps)

      # -------- Reverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t

        _x = x
        x, x_mean = corrector_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = corrector_obj_adj.update_fn(_x, adj, flags, vec_t)

        _x = x
        x, x_mean = predictor_obj_x.update_fn(x, adj, flags, vec_t)
        adj, adj_mean = predictor_obj_adj.update_fn(_x, adj, flags, vec_t)
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), diff_steps * (n_steps + 1)
  return pc_sampler


# -------- S4 solver --------
def S4_solver(sde_x, sde_adj, shape_x, shape_adj, predictor='None', corrector='None',
                        snr=0.1, scale_eps=1.0, n_steps=1,
                        probability_flow=False, continuous=False,
                        denoise=True, eps=1e-3, device='cuda'):

  def s4_solver(model_x, model_adj, init_flags, diff_steps=None):

    score_fn_x = get_score_fn(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn(sde_adj, model_adj, train=False, continuous=continuous)

    with torch.no_grad():
      # -------- Initial sample --------
      x = sde_x.prior_sampling(shape_x).to(device)
      adj = sde_adj.prior_sampling_sym(shape_adj).to(device)
      flags = init_flags
      x = mask_x(x, flags)
      adj = mask_adjs(adj, flags)
      if diff_steps is None:
        diff_steps = sde_adj.N

      timesteps = torch.linspace(sde_adj.T, eps, diff_steps, device=device)
      dt = -1. / diff_steps

      # -------- Rverse diffusion process --------
      for i in trange(0, (diff_steps), desc = '[Sampling]', position = 1, leave=False):
        t = timesteps[i]
        vec_t = torch.ones(shape_adj[0], device=t.device) * t
        vec_dt = torch.ones(shape_adj[0], device=t.device) * (dt/2)

        # -------- Score computation --------
        score_x = score_fn_x(x, adj, flags, vec_t)
        score_adj = score_fn_adj(x, adj, flags, vec_t)

        Sdrift_x = -sde_x.sde(x, vec_t)[1][:, None, None] ** 2 * score_x
        Sdrift_adj  = -sde_adj.sde(adj, vec_t)[1][:, None, None] ** 2 * score_adj

        # -------- Correction step --------
        timestep = (vec_t * (sde_x.N - 1) / sde_x.T).long()

        noise = gen_noise(x, flags, sym=False)
        grad_norm = torch.norm(score_x.reshape(score_x.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_x, VPSDE):
          alpha = sde_x.alphas.to(vec_t.device)[timestep]
        else:
          alpha = torch.ones_like(vec_t)

        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        x_mean = x + step_size[:, None, None] * score_x
        x = x_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        noise = gen_noise(adj, flags)
        grad_norm = torch.norm(score_adj.reshape(score_adj.shape[0], -1), dim=-1).mean()
        noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
        if isinstance(sde_adj, VPSDE):
          alpha = sde_adj.alphas.to(vec_t.device)[timestep] # VP
        else:
          alpha = torch.ones_like(vec_t) # VE
        step_size = (snr * noise_norm / grad_norm) ** 2 * 2 * alpha
        adj_mean = adj + step_size[:, None, None] * score_adj
        adj = adj_mean + torch.sqrt(step_size * 2)[:, None, None] * noise * scale_eps

        # -------- Prediction step --------
        x_mean = x
        adj_mean = adj
        mu_x, sigma_x = sde_x.transition(x, vec_t, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t, vec_dt)
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

        x = x + Sdrift_x * dt
        adj = adj + Sdrift_adj * dt

        mu_x, sigma_x = sde_x.transition(x, vec_t + vec_dt, vec_dt)
        mu_adj, sigma_adj = sde_adj.transition(adj, vec_t + vec_dt, vec_dt)
        x = mu_x + sigma_x[:, None, None] * gen_noise(x, flags, sym=False)
        adj = mu_adj + sigma_adj[:, None, None] * gen_noise(adj, flags)

        x_mean = mu_x
        adj_mean = mu_adj
      print(' ')
      return (x_mean if denoise else x), (adj_mean if denoise else adj), 0
  return s4_solver


from dpm_solver_joint_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver_Joint
def get_dpm_solver(sde_x, sde_adj, shape_x, shape_adj, predictor='Euler', corrector='None',
                   snr=0.1, scale_eps=1.0, n_steps=1,
                   probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', dpm_config={}):

  def sampler(model_x, model_adj, init_flags, diff_steps=None):

    score_fn_x = get_score_fn_dpm_compt(sde_x, model_x, train=False, continuous=continuous)
    score_fn_adj = get_score_fn_dpm_compt(sde_adj, model_adj, train=False, continuous=continuous)
    corrector_fn = LangevinCorrector if corrector=='Langevin' else NoneCorrector
    snr = 0.2
    scale_eps = 0.7
    n_steps = 1
    corrector_obj_x = corrector_fn('x', sde_x, score_fn_x, snr, scale_eps, n_steps)
    corrector_obj_y = corrector_fn('adj', sde_adj, score_fn_adj, snr, scale_eps, n_steps)

    corrector_x = None
    corrector_y = None
    if dpm_config["use_corrector"]:
      corrector_x = corrector_obj_x.update_fn
      corrector_y = corrector_obj_y.update_fn

    x_T = sde_x.prior_sampling(shape_x).to(device)
    adj_T = sde_adj.prior_sampling_sym(shape_adj).to(device)
    flags = init_flags
    x_T = mask_x(x_T, flags)
    adj_T = mask_adjs(adj_T, flags)


    noise_schedule_x = NoiseScheduleVP(schedule='linear', continuous_beta_0=dpm_config['continuous_beta_0_x'], continuous_beta_1=dpm_config['continuous_beta_1_x'])
    noise_schedule_adj = NoiseScheduleVP(schedule='linear', continuous_beta_0=dpm_config['continuous_beta_0_adj'], continuous_beta_1=dpm_config['continuous_beta_1_adj'])

    model_fn_x = model_wrapper(
        sde_x,
        model_x,
        noise_schedule_x,
        model_type=dpm_config["model_type"],  # or "x_start" or "v" or "score"
        model_kwargs={'flags': flags},
    )

    model_fn_adj = model_wrapper(
        sde_adj,
        model_adj,
        noise_schedule_adj,
        model_type=dpm_config["model_type"],  # or "noise", "x_start" or "v" or "score"
        model_kwargs={'flags': flags},
    )

    dpm_solver = DPM_Solver_Joint(model_fn_x, model_fn_adj, noise_schedule_x, noise_schedule_adj) #, algorithm_type="dpmsolver++")

    x_sample, adj_sample = dpm_solver.sample_joint(
      x_T,
      adj_T,
      t_start=sde_x.T,  # FIXME
      t_end=dpm_config["t_end"],
      steps=diff_steps,
      order=dpm_config["order"],
      skip_type=dpm_config["skip_type"],
      method=dpm_config["method"],
      corrector_fn_x=corrector_x,
      corrector_fn_y=corrector_y,
      y_shape_0=shape_adj[0],
      p_model_kwargs={'flags': flags},
      denoise_to_zero=dpm_config["denoise_to_zero"],
      atol=dpm_config["atol"],
      rtol=dpm_config["rtol"],
      lower_order_final=False,
    )

    return x_sample, adj_sample, 0

  return sampler
