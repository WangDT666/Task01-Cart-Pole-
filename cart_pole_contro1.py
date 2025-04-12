"""
姓名：    王帅
学号：241404010227
微信：wangdaotiandaowang
手机：13213612963
"""
import gym
import numpy as np

# 初始化环境（指定渲染模式）
env = gym.make('CartPole-v1', render_mode="human")  # 使用 render_mode="human"

# PID控制器参数
Kp = 2.0             #控制响应速度
Ki = 0.02            #消除稳态误差
Kd = 0.5             #抑制震荡
"""
调参逻辑
Kp：初始值设为1.5，观察系统响应。如果振荡严重，减小Kp；如果响应太慢，增大Kp。
Ki：初始值设为0.01，观察稳态误差。如果误差无法消除，增大Ki；如果系统过调，减小Ki。
Kd：初始值设为0.5，观察振荡情况。如果振荡严重，增大Kd；如果响应太慢，减小Kd。
"""
for episode in range(200):
    observation, info = env.reset()     # 重置环境，返回初始状态和附加信息 observation 和 info

    integral = 0.0                      #积分项的累加值
    previous_error = 0.0                #存储上一次的误差（用于计算微分项）
    total_reward = 0                    #累计奖励（CartPole每存活一步奖励+1）
    dt = 0.02                           # 时间步长
    done = False                        #标记当前Episode是否结束（杆子倒下或超时）

    while not done:

        # 提取状态（4个值）
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = observation
        #小车位置（范围 [-4.8, 4.8]），小车速度，杆子与竖直方向夹角（弧度），杆子角速度=环境返回的4维状态数组

        # 计算PID控制信号
        error = pole_angle                      #杆子角度偏差（目标为0，即竖直）
        P=Kp * error

        integral += error * dt
        integral = max(min(integral, 2.0), -2.0)  # 限制积分范围，防止过大
        I = Ki * integral

        derivative = (error - previous_error) / dt  # 误差变化率（通过前后两次误差计算）
        D = Kd * derivative

        pid_output=P+I+D

        previous_error = error
        action = 1 if pid_output > 0 else 0

        # 执行动作（适配新版step()的5个返回值）
        observation, reward, terminated, truncated, info = env.step(action)
        #新的状态，即时奖励（此处固定为1，存活一步奖励+1），任务失败（如杆子倒下），超时终止（默认500步后结束），附加调试信息，

        done = terminated or truncated      #若为 True，结束当前Episode
        total_reward += reward

        env.render()  # 不再触发警告，根据 render_mode 显示图形界面

    print(f"Episode {episode + 1}, Reward: {total_reward}") #打印每个Episode的总奖励（即存活步数）

env.close()