import os
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import torch
import imutil
import time

from env import Env


# Globals
Ts, rewards, Qs, best_avg_reward = [], [], [], -1e10


# Test DQN
def test(args, T, dqn, val_mem, evaluate=False):
  global Ts, rewards, Qs, best_avg_reward
  env = Env(args)
  env.eval()
  Ts.append(T)
  T_rewards, T_Qs = [], []
  if args.render_video:
    filename = 'evaluation_{}.mp4'.format(int(time.time()))
    vid = imutil.Video(filename)

  # Test performance over several episodes
  done = True
  for _ in range(args.evaluation_episodes):
    t = 0
    while True:
      t += 1
      if done:
        state, reward_sum, done = env.reset(), 0, False

      action = dqn.act_e_greedy(state)  # Choose an action ε-greedily
      state, reward, done = env.step(action)  # Step
      reward_sum += reward
      if args.render:
        env.render()
      if args.render_video:
        caption = "t={} reward={} action={}".format(t, reward_sum, action)
        vid.write_frame(state[-1], resize_to=(512,512), caption=caption)

      if done:
        T_rewards.append(reward_sum)
        break
  env.close()
  if args.render_video:
    vid.finish()

  # Test Q-values over validation memory
  for state in val_mem:  # Iterate over valid states
    T_Qs.append(dqn.evaluate_q(state))

  avg_reward, avg_Q = sum(T_rewards) / len(T_rewards), sum(T_Qs) / len(T_Qs)
  if not evaluate:
    # Append to results
    rewards.append(T_rewards)
    Qs.append(T_Qs)

    # Plot
    _plot_line(Ts, rewards, 'Reward', path='results')
    _plot_line(Ts, Qs, 'Q', path='results')

    # Save model parameters if improved
    if avg_reward > best_avg_reward:
      best_avg_reward = avg_reward
      dqn.save('results')

  # Return average reward and Q-value
  return avg_reward, avg_Q


# Plots min, max and mean + standard deviation bars of a population over time
def _plot_line(xs, ys_population, title, path=''):
  max_colour, mean_colour, std_colour, transparent = 'rgb(0, 132, 180)', 'rgb(0, 172, 237)', 'rgba(29, 202, 255, 0.2)', 'rgba(0, 0, 0, 0)'

  ys = torch.tensor(ys_population, dtype=torch.float32)
  ys_min, ys_max, ys_mean, ys_std = ys.min(1)[0].squeeze(), ys.max(1)[0].squeeze(), ys.mean(1).squeeze(), ys.std(1).squeeze()
  ys_upper, ys_lower = ys_mean + ys_std, ys_mean - ys_std

  trace_max = Scatter(x=xs, y=ys_max.numpy(), line=Line(color=max_colour, dash='dash'), name='Max')
  trace_upper = Scatter(x=xs, y=ys_upper.numpy(), line=Line(color=transparent), name='+1 Std. Dev.', showlegend=False)
  trace_mean = Scatter(x=xs, y=ys_mean.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=mean_colour), name='Mean')
  trace_lower = Scatter(x=xs, y=ys_lower.numpy(), fill='tonexty', fillcolor=std_colour, line=Line(color=transparent), name='-1 Std. Dev.', showlegend=False)
  trace_min = Scatter(x=xs, y=ys_min.numpy(), line=Line(color=max_colour, dash='dash'), name='Min')

  plotly.offline.plot({
    'data': [trace_upper, trace_mean, trace_lower, trace_min, trace_max],
    'layout': dict(title=title, xaxis={'title': 'Step'}, yaxis={'title': title})
  }, filename=os.path.join(path, title + '.html'), auto_open=False)
