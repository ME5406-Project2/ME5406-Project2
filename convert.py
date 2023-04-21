import json
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from TensorBoard JSON file
with open('/Users/praveenelango/ME5406-Project2/trained_models/training70_rand_CPG/SAC_ent_coef.json', 'rb') as f:
    data = json.load(f)

sns.set_style("darkgrid")
# Extract training data from the JSON file
train_steps = []
train_rewards = []
# print(data)
# for event in data:
#     if 'summary' in event:
#         for value in event['summary']['value']:
#             if value['tag'] == 'eval/mean_reward':
#                 train_steps.append(event['step'])
#                 train_rewards.append(value['simple_value'])

for event in data:
    train_steps.append(event[1])
    train_rewards.append(event[2])

# Apply exponential smoothing with a smoothing scalar of 0.99
smoothed_rewards = []
previous = train_rewards[0]
for reward in train_rewards:
    smoothed_reward = previous * 0.95 + reward * 0.05
    smoothed_rewards.append(smoothed_reward)
    previous = smoothed_reward

# Plot the smoothed rewards over training steps
plt.plot(train_steps, smoothed_rewards)
plt.title('SAC Entropy Coefficient Over Time')
plt.xlabel('Training Steps')
plt.ylabel('Entropy Coefficient')
plt.show()
