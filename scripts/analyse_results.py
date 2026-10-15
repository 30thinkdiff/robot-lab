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


# def test(self, file_name: str):
#     filepath = os.path.join("..", "data", file_name)

#     validnum = 0
#     successnum = 0
#     succesrate = 0
#     result = {"total":0, "success":0, "successrate":0, "error":""}
#     try:
#         with  open(file=filepath, mode="r", encoding="utf-8") as f:
#             data = csv.reader(f)
#             try:
#                 next(data)
#             except StopIteration:
#                 print(f"\n{file_name} 文件为空")
#                 result["error"] = "文件为空"
#                 return result

#             for row in data:
#                 if row[1] in ("0", "1"):
#                     validnum += 1
#                     if row[1] == "1":
#                         successnum += 1
#     except FileNotFoundError:
#         print("文件不存在")
#         result["error"] = "文件不存在"
#         return result
        
#     print(f"\ntotal episodes:{validnum}")
#     print(f"success episodes:{successnum}")

#     if validnum:
#         succesrate = round(successnum/validnum, 2)
#         print(f"Scuccess rate:{succesrate}")
#     else:
#         succesrate = "no valid episodes"
#     result["total"] = validnum
#     result["success"] = successnum
#     result["successrate"] = succesrate
#     return result
# 参考 https://blog.csdn.net/a5225354/article/details/106356770
# import unittest

# class Test(unittest.TestCase):
#     def test_add_01(self):
#         result = test(self, "example_eval.csv")
#         self.assertEqual(result["total"], 3)
#         self.assertEqual(result["success"], 2)
#         self.assertEqual(result["successrate"], 0.67)

#     def test_add_02(self):
#         result = test(self, "example_eval1.csv")
#         self.assertEqual(result["total"], 3)
#         self.assertEqual(result["success"], 0)
#         self.assertEqual(result["successrate"], 0.00)


#     def test_add_03(self):
#         result = test(self, "example_eval2.csv")
#         self.assertEqual(result["error"], "文件为空")

#     def test_add_04(self):
#         result = test(self, "example_eval3.csv")
#         self.assertEqual(result["error"], "文件不存在")

# if __name__ == "__main__":
#     suite = unittest.TestSuite()
#     suite.addTest(Test("test_add_01"))
#     suite.addTest(Test("test_add_02"))
#     suite.addTest(Test("test_add_03"))
#     suite.addTest(Test("test_add_04"))
#     runner = unittest.TextTestRunner()
#     runner.run(suite)

    
## GLM5.1给的参考答案
import csv
import os
import unittest
from typing import Dict, Union

class CSVAnalyzer:
    """CSV数据分析器：用于统计实验的成功率等信息"""
    
    @staticmethod
    def analyse(file_name: str) -> Dict[str, Union[int, float, str]]:
        """
        分析指定的CSV文件并返回统计结果。
        
        Args:
            file_name (str): 文件名（不含路径）
            
        Returns:
            Dict: 包含 total, success, successrate, error 的字典
        """

        SUCCESS_COL_NAME = "success"
        VALID_VALUES = {"0", "1"} # 使用集合查找速度更快

        filepath = os.path.join("..", "data", file_name)
        
        # 初始化默认返回结果，明确类型
        result = {
            "total": 0, 
            "success": 0, 
            "successrate": 0.0, 
            "error": ""
        }
        
        # 1. 检查文件是否存在
        if not os.path.exists(filepath):
            result["error"] = "文件不存在"
            return result

        try:
            with open(file=filepath, mode="r", encoding="utf-8") as f:
                data = csv.reader(f)
                
                # 2. 读取表头，动态获取目标列的索引（比写死 row[1] 更健壮）
                try:
                    header = next(data)
                except StopIteration:
                    result["error"] = "文件为空"
                    return result
                
                # 防御性编程：如果表头里没有 success 列，直接报错
                if SUCCESS_COL_NAME not in header:
                    result["error"] = "缺少success列"
                    return result
                
                success_idx = header.index(SUCCESS_COL_NAME)

                # 3. 核心计算逻辑（纯计算，没有任何 print）
                validnum = 0
                successnum = 0
                
                for row in data:
                    # 防止某一行数据残缺导致 IndexError
                    if len(row) <= success_idx:
                        continue
                        
                    val = row[success_idx]
                    if val in VALID_VALUES:
                        validnum += 1
                        if val == "1":
                            successnum += 1
                            
        except Exception as e:
            # 捕获其他未知异常，如权限不足等
            result["error"] = f"读取文件时发生错误: {str(e)}"
            return result

        # 4. 组装结果
        result["total"] = validnum
        result["success"] = successnum
        if validnum > 0:
            result["successrate"] = round(successnum / validnum, 2)
        else:
            result["error"] = "no valid episodes"
            
        return result


# ==================== 测试代码 ====================

class TestCSVAnalyzer(unittest.TestCase):
    """CSVAnalyzer的单元测试类"""
    
    def test_normal_file(self):
        """测试正常的CSV文件"""
        result = CSVAnalyzer.analyse("example_eval.csv")
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["success"], 2)
        self.assertEqual(result["successrate"], 0.67)
        self.assertEqual(result["error"], "")

    def test_all_fail_file(self):
        """测试成功率为0的CSV文件"""
        result = CSVAnalyzer.analyse("example_eval1.csv")
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["success"], 0)
        self.assertEqual(result["successrate"], 0.0)

    def test_empty_file(self):
        """测试空文件"""
        result = CSVAnalyzer.analyse("example_eval2.csv")
        self.assertEqual(result["error"], "文件为空")

    def test_not_exist_file(self):
        """测试不存在的文件"""
        result = CSVAnalyzer.analyse("example_eval3.csv")
        self.assertEqual(result["error"], "文件不存在")

if __name__ == "__main__":
    # 更简洁的运行测试的方式
    unittest.main()
