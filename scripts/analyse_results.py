## 读取CSV文件
## 统计成功次数
## 计算成功率

## 不会写，参考https://blog.csdn.net/weixin_46124984/article/details/124655560

# import csv

# csv_reader = csv.reader(open("../data/example_eval.csv"))
# data = []
# for row in csv_reader:
#     data.append(row)
#     print(row)
#     print(data)
# validnum = 0
# successnum = 0
# for i in range(len(data)):
#     print(type(data[i][1]))
#     if data[i][1] == "0" or data[i][1] == "1":
#         validnum += 1
#         if data[i][1] == "1":
#             successnum += 1
# print(validnum)
# if validnum:
#     print(f"{successnum/validnum:.2f}")

# print(csv_reader) 

## 第二版
import csv
import os

# filepath = os.path.join("..", "data", "example_eval5.csv")

# validnum = 0
# successnum = 0

# with  open(file=filepath, mode="r", encoding="utf-8") as f:
#     data = csv.reader(f)
#     try:
#         next(data)
#     except StopIteration:
#         print("文件为空")

#     for row in data:
#         if row[1] in ("0", "1"):
#             validnum += 1
#             if row[1] == "1":
#                 successnum += 1

# print(f"total episodes:{validnum}")
# print(f"success episodes:{successnum}")

# if validnum:
#     print(f"Scuccess rate:{successnum/validnum:.2f}")


def test(self, file_name: str):
    filepath = os.path.join("..", "data", file_name)

    validnum = 0
    successnum = 0
    try:
        with  open(file=filepath, mode="r", encoding="utf-8") as f:
            data = csv.reader(f)
            try:
                next(data)
            except StopIteration:
                print(f"\n{file_name} 文件为空")
                return 

            for row in data:
                if row[1] in ("0", "1"):
                    validnum += 1
                    if row[1] == "1":
                        successnum += 1
    except FileNotFoundError:
        print("文件不存在")
        
    print(f"\ntotal episodes:{validnum}")
    print(f"success episodes:{successnum}")

    if validnum:
        print(f"Scuccess rate:{successnum/validnum:.2f}")
# 参考 https://blog.csdn.net/a5225354/article/details/106356770
import unittest

class Test(unittest.TestCase):
    def test_add_01(self):
        test(self, "example_eval.csv")

    def test_add_02(self):
        test(self, "example_eval1.csv")

    def test_add_03(self):
        test(self, "example_eval2.csv")

    def test_add_04(self):
        test(self, "example_eval3.csv")

if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(Test("test_add_01"))
    suite.addTest(Test("test_add_02"))
    suite.addTest(Test("test_add_03"))
    suite.addTest(Test("test_add_04"))
    runner = unittest.TextTestRunner()
    runner.run(suite)

    
