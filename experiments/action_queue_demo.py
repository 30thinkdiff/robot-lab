from collections import deque
import unittest

## 参考博客https://blog.csdn.net/chl183/article/details/106958004
class SimplePolicy:
    def __init__(self, predict_fn, n_action_steps):
        if n_action_steps <= 0:
            raise ValueError("n_action_steps 必须大于 0")
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
            if len(tmp) < self.n_action_steps:
                raise ValueError("预测函数返回的动作数量不足")
            else:
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

class TestSimplePolicy(unittest.TestCase):
    def test_normal(self):
        policy = SimplePolicy(
            predict_fn=fake_predict,
            n_action_steps=2
        )

        # 第一次调用，应该触发预测，返回第一个动作
        action1 = policy.select_action(1)
        self.assertEqual(action1, 100)  # 应该是 100
        
        # 第二次调用，应该从队列中取出第二个动作
        action2 = policy.select_action(1)
        self.assertEqual(action2, 101)  # 应该是 101
        
        # 第三次调用，队列空了，应该再次触发预测
        action3 = policy.select_action(1)
        self.assertEqual(action3, 100)  # 应该是新预测的第一个动作
        
        # 第四次调用，应该从队列中取出第二个动作
        action4 = policy.select_action(1)
        self.assertEqual(action4, 101)  # 应该是新预测的第二个动作

    def test_overlimiterror(self):
        policy = SimplePolicy(
            predict_fn=fake_predict,
            n_action_steps=7
        )
        with self.assertRaises(ValueError):
            policy.select_action(7)

    def test_negtiveerror(self):
        with self.assertRaises(ValueError):
            SimplePolicy(predict_fn=fake_predict, n_action_steps=-1)
if __name__ == "__main__":

    policy = SimplePolicy(
        predict_fn=fake_predict,
        n_action_steps=2
    )

    print(policy.select_action(1))
    # print(policy.select_action(2))
    # print(policy.select_action(3))

    # print("模型调用次数:", call_count)

    # policy.reset()

    # print(policy.select_action(4))

    # print("模型调用次数:", call_count)
    unittest.main()