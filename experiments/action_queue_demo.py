from collections import deque

## 参考博客https://blog.csdn.net/chl183/article/details/106958004
class SimplePolicy:
    def __init__(self, predict_fn, n_action_steps):
        self.predict_fn = predict_fn
        self.n_action_steps = n_action_steps
        self.queue = deque()

    def reset(self):
        # TODO 1：清空已有动作队列
        self.queue.clear()

    def select_action(self, observation):
        # TODO 2：如果队列为空，就调用预测函数
        # 得到动作块，并取前 n_action_steps 个
        # 放进队列
        if not self.queue:
            tmp = self.predict_fn(observation)
            self.queue.extend(tmp[:self.n_action_steps])

        return self.queue.popleft()

# ===== 用假模型测试，不需要 GPU =====

call_count = 0


def fake_predict(observation):
    global call_count
    call_count += 1

    # 每次产生五个假动作
    return [
        observation * 100 + i
        for i in range(5)
    ]


policy = SimplePolicy(
    predict_fn=fake_predict,
    n_action_steps=2
)

print(policy.select_action(1))
print(policy.select_action(2))
print(policy.select_action(3))

print("模型调用次数:", call_count)

policy.reset()

print(policy.select_action(4))

print("模型调用次数:", call_count)
