# ==============================================================================
# DEEPDRIFT RL: SPATIAL FRICTION VS ACTION ENTROPY
# ==============================================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import PPO
from deepdrift import DeepDriftMonitor
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# 1. CONFIG
class Config:
    ENV_ID = "CartPole-v1"
    TIMESTEPS = 15000
    N_EPISODES = 50
    # Stronger stress to provoke internal friction
    SENSOR_NOISE = 0.5
    SENSOR_SCALE = 1.8 

# 2. TRAINING
print("🔄 Training PPO agent...")
env = gym.make(Config.ENV_ID)
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=Config.TIMESTEPS)

# 3. MONITORING SPATIAL FRICTION
# We monitor all layers of the policy network to catch "Information Friction"
# In SB3, layers are policy_net[0], policy_net[2], etc.
target = model.policy.mlp_extractor.policy_net
monitor = DeepDriftMonitor(target, layer_names=None, pooling='flatten')

# 4. ENGINE
def run_eval(env, model, monitor, is_stress=False):
    res = {'frictions': [], 'entropies': [], 'rewards': []}
    
    for _ in tqdm(range(Config.N_EPISODES), desc="Testing"):
        obs, _ = env.reset()
        done = False
        frics, ents, total_rew = [], [], 0
        
        while not done:
            if is_stress:
                obs = obs * Config.SENSOR_SCALE + np.random.normal(0, Config.SENSOR_NOISE, size=obs.shape)
            
            obs_t = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)
            
            # --- DEEPDRIFT: SPATIAL VELOCITY (FRICTION) ---
            monitor.clear()
            _ = model.policy.mlp_extractor(obs_t)
            # We sum velocities between ALL internal layers
            v = monitor.get_spatial_velocity()
            frics.append(np.sum(v)) 
            
            # --- SOTA: ENTROPY ---
            dist = model.policy.get_distribution(obs_t)
            ents.append(dist.entropy().item())
            
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_rew += reward
            done = terminated or truncated
            
        res['rewards'].append(total_rew)
        res['frictions'].append(np.mean(frics))
        res['entropies'].append(np.mean(ents))
        
    return {k: np.array(v) for k, v in res.items()}

# 5. EXECUTION
res_id = run_eval(env, model, monitor, is_stress=False)
res_ood = run_eval(env, model, monitor, is_stress=True)

# 6. ANALYSIS
y_true = np.concatenate([np.zeros(Config.N_EPISODES), np.ones(Config.N_EPISODES)])
def get_auroc(id_v, ood_v):
    scores = np.concatenate([id_v, ood_v])
    score = roc_auc_score(y_true, scores)
    return max(score, 1 - score)

auroc_dd = get_auroc(res_id['frictions'], res_ood['frictions'])
auroc_sota = get_auroc(res_id['entropies'], res_ood['entropies'])

# 7. PLOT
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.hist(res_id['frictions'], bins=15, alpha=0.5, label='Clean', color='blue')
plt.hist(res_ood['frictions'], bins=15, alpha=0.5, label='Stress (OOD)', color='red')
plt.title("Information Friction Distribution"); plt.legend()

plt.subplot(1, 2, 2)
plt.bar(['Entropy (SOTA)', 'DeepDrift (Friction)'], [auroc_sota, auroc_dd], color=['gray', 'orange'])
plt.ylim(0.4, 1.05); plt.title("Detection Accuracy (AUROC)")
for i, v in enumerate([auroc_sota, auroc_dd]): plt.text(i, v+0.01, f'{v:.4f}', ha='center')
plt.show()

print(f"Final AUROC - DeepDrift: {auroc_dd:.4f} | Entropy: {auroc_sota:.4f}")
monitor.remove_hooks()
