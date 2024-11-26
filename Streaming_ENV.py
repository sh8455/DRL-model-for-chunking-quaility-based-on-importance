import numpy as np
import random
import matplotlib.pyplot as plt

from stable_baselines3 import DQN

# Defining the Video Streaming Environment Class
class VideoStreamingEnv:
    def __init__(self, bandwidths, chunk_play_time=4, user_bandwidth = 3):
        self.num_chunks = 100 # 청크 개수
        self.bandwidths = bandwidths # 대역폭 종류 (리스트)
        self.qualities = [1, 2, 3, 4, 5, 6] # 선택 가능한 화질의 종류 (리스트)
        self.chunk_important = np.array([random.randint(1, 6) for _ in range(self.num_chunks)]) # 청크별 중요도 (np.array)
        self.current_chunk = 0 # 현재 청크의 순서를 알기 위한 변수
        self.chunk_play_time = chunk_play_time # 청크별 재생 시간 (4초로 일정하다고 가정)
        self.user_bandwidth = user_bandwidth
        
        self.chunk_qualities = [] # 설정된 청크별 화질을 담을 변수
        self.chunk_latency = [] # 청크별 버퍼링 시간을 담을 변수
        
    def reset(self):
        self.chunk_important = np.array([random.randint(1,6) for _ in range(self.num_chunks)])
        self.current_chunk = 0
        self.chunk_qualities = []
        self.chunk_latency = []
        return self._get_state()
        
    # reward를 계산하는 함수
    def caculate_rewards(self, max_limit):
        cur_important = self.chunk_important[self.current_chunk] # 현재 청크의 중요도
        cur_quality = self.chunk_qualities[self.current_chunk] # 현재 스텝에서 설정된 화질
        cur_buffering = self.chunk_latency[self.current_chunk] # 현재 스텝에서 설정된 버퍼링 시간

        prev_quality = self.chunk_qualities[self.current_chunk - 1] if self.current_chunk > 0 else 0 # 한 단계 전 스텝에서 설정된 화질
        prev_buffering = self.chunk_latency[self.current_chunk - 1] if self.current_chunk > 0 else 0 # 한 단계 전 스텝에서 설정된 버퍼링 시간

        quality_diff = -abs(prev_quality - cur_quality) # 한 단계 전 화질과 현 화질의 차이
    
        buffer_diff = prev_buffering - cur_buffering # 한 단계 전 지연시간과 현 지연시간의 차이
        
        chunk_important_quality = 0 if cur_quality > max_limit else cur_important * cur_quality # 청크별 중요도와 현재 화질의 곱

        qoe = (cur_quality + quality_diff + buffer_diff + chunk_important_quality)
        
        # 총 청크별 화질의 합 제한
        done = False
        if cur_quality > max_limit:
            done = True
            qoe = -50

        print("max limit:", max_limit, "cur importance:", cur_important, "cur quality:", cur_quality, "cur buffering:", cur_buffering, "chunk importance quality:", chunk_important_quality, "qoe:", qoe)
        return qoe, done
    
    # 화질 범위를 계산
    def set_quality_range(self):
        # 중요도의 최소값과 최대값을 계산
        min_importance = min(self.chunk_important)
        max_importance = max(self.chunk_important)
        
        # 사용자의 제한 화질을 계산하고, 가능한 최소 화질과 최대 화질을 설정
        max_limit = self.set_limit_quality()
        min_quality = 1
        max_quality = max_limit
        
        print("min_quality:", min_quality, "max_quality:", max_quality)
        return min_importance, max_importance, min_quality, max_quality
    
    def map_importance_to_quality(self, importance):
        # 중요도와 화질의 범위를 가져옴
        min_importance, max_importance, min_quality, max_quality = self.set_quality_range()
        
        # 선형 변환을 사용하여 중요도가 최소일 때 최소 화질에, 중요도가 최대일 때 최대 화질에 매핑되도록 함
        assigned_quality = min_quality + (max_quality - min_quality) * (importance - min_importance) / (max_importance - min_importance) 
        
        # 매핑된 화질을 반올림하여 정수로 반환
        return int(round(assigned_quality))
    
    # 사용자 대역폭별로 제안 화질을 설정
    def set_limit_quality(self):
        user_bandwidth = self.user_bandwidth
        max_limit = None

        if user_bandwidth == 3:
            # max_limit = 360
            max_limit = 2
            # self.limit = 25
        elif user_bandwidth == 5:
            # max_limit = 480
            max_limit = 3
            # self.limit = 32
        elif user_bandwidth == 10:
            # max_limit = 720
            max_limit = 4
            # self.limit = 37
        elif user_bandwidth == 15:
            # max_limit = 1080
            max_limit = 5
            # self.limit = 40
        else:
            # max_limit = 1440
            max_limit = 6
            # self.limit = 41

        return max_limit
    
    def step(self, action):
        max_limit = self.set_limit_quality() # 사용자의 제한 범위 화질, limits
        cur_important = self.chunk_important[self.current_chunk]
        
        min_quality = 1
        quality_range = self.map_importance_to_quality(cur_important)
        print("quality range:", quality_range)
        
        if self.qualities[action] <= quality_range:
            chunk_quality = self.qualities[action] 
        else:
            chunk_quality = quality_range # 청크별 화질을 설정
            reward = -10
        #chunk_quality = random.randint(1, 6) # 환경 테스트를 위해 넣은 랜덤함수
        self.chunk_qualities.append(chunk_quality)

        if chunk_quality > max_limit: # 만약 설정된 화질이 제한 범위보다 크다면
            chunk_play_t = self.chunk_play_time + (chunk_quality - max_limit)
        else:
            chunk_play_t = self.chunk_play_time

        self.chunk_latency.append(chunk_play_t)

        reward, done = self.caculate_rewards(max_limit)
        
        if done == False:
            self.current_chunk += 1
            done = self.current_chunk >= self.num_chunks

        return self._get_state(), reward, done, {}

    def _get_state(self):
        cur_quality = self.chunk_qualities[self.current_chunk] if self.current_chunk < len(self.chunk_qualities) else 0
        cur_buffering = self.chunk_latency[self.current_chunk] if self.current_chunk < len(self.chunk_latency) else 0
        prev_quality = self.chunk_qualities[self.current_chunk - 1] if self.current_chunk > 0 else 0
        prev_buffering = self.chunk_latency[self.current_chunk - 1] if self.current_chunk > 0 else 0
        obs = np.concatenate((np.array([self.user_bandwidth, prev_quality, prev_buffering, cur_quality, cur_buffering]), self.chunk_important))
        return obs
    
bandwidths = [3, 5, 10, 15, 20]

env = VideoStreamingEnv(bandwidths, chunk_play_time=4, user_bandwidth=20)

model = DQN.load("dqn_v4")

num_episode = 5
rewards_per_episode = []
chunk_important_per_episode = []
chunk_qualities_per_episode = []

model1_chunk_important_list = []

for episode in range(num_episode):
    obs = env.reset()
    model1_chunk_important_list.append(env.chunk_important.copy())
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        # episode_chunk_qualities.append(env.chunk_qualities[-1])
    
    rewards_per_episode.append(total_reward)
    chunk_important_per_episode.append(env.chunk_important.copy())
    chunk_qualities_per_episode.append(env.chunk_qualities.copy())
    print(f"Episode {episode + 1}: Total reward = {total_reward}")

    print("\nRewards per episode:", rewards_per_episode)
    
np.save('model1_chunk_important_list.npy', model1_chunk_important_list)
np.save('model1_chunk_qualities_per_episode_max2.npy', chunk_qualities_per_episode)
    
# 시각화
for episode in range(num_episode):
    plt.figure(figsize=(10, 6))
    plt.plot(chunk_important_per_episode[episode], label ="Chunk Importance")
    plt.plot(chunk_qualities_per_episode[episode], label ="Chunk Qualities")
    plt.xlabel("Chunk")
    plt.ylabel("Value")
    plt.title(f"Episode {episode + 1}")
    plt.legend()
    plt.show()
