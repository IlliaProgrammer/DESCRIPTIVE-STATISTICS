import math

import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.ticker import PercentFormatter
import pandas as pd


variant = 120 % 14 + 14 * 8
n = 125
sigma = 1.9


np.random.seed(42)
sample = np.random.normal(0, sigma, n).round(4)


math_expectations = np.random.dirichlet(np.ones(n), size=1).round(4)

sample_dict = dict(zip(sample, math_expectations[0]))

od = collections.OrderedDict(sorted(sample_dict.items()))
odv = dict(sorted(od.items(), key=lambda item: item[1], reverse=True))



def print_dict(d:dict):
    print("{:<10}  {:<10}".format("x", "probability"))
    for x, probability in d.items():
        print("{:<10}  {:<10}".format(x, probability))



def calculate_mean(distribution_dict: dict) -> float:
    mean = 0
    for value, probability in distribution_dict.items():
        mean += value * probability
    return mean



def calc_sample_mean(distribution_dict: dict) -> float:
    sample_mean = 0
    for v, probability in distribution_dict.items():
        sample_mean += v
    sample_mean *= 1 / len(distribution_dict)
    return sample_mean



def calc_median(distribution_dict: dict) -> float:
    sorted_x = sorted(distribution_dict.keys())
    length = len(sorted_x)
    if length % 2 == 0:
        median = (sorted_x[round(length / 2)] + sorted_x[round(length / 2) + 1]) / 2
    else:
        median = sorted_x[round(length / 2)]
    return median



def calc_mode(distribution_dict: dict) -> list:
    mode = []
    list_numbers = {}

    for k in distribution_dict.keys():
        values = list(distribution_dict.keys())
        counter = values.count(k)
        list_numbers.update({k: counter})

    max_counter = max(list_numbers.values())
    for k, v in list_numbers.items():
        if v == max_counter:
            mode.append(k)

    return mode



def calc_variance(distribution_dict: dict) -> float:
    sample_mean = calc_sample_mean(distribution_dict)
    variance = 0
    for x in distribution_dict.keys():
        variance += (x - sample_mean) ** 2
    variance /= len(distribution_dict) - 1
    return variance



def calc_standard_deviation(distribution_dict: dict) -> float:
    variance=calc_variance(distribution_dict)
    return math.sqrt(variance)



plt.plot(od.keys(), od.values())
plt.suptitle("Полігон")


# fig = plt.figure()
# ax = fig.add_subplot (111)
# ax.hist (sample, edgecolor='black')
# plt.suptitle("Гістограма")


# plt.boxplot(od.keys(), od.values())
# plt.suptitle("Розмаху")


# df = pd.DataFrame.from_dict({'value': odv.values()})

# df = df.sort_values(by='value',ascending=False)
# df["cumpercentage"] = df["value"].cumsum()/df["value"].sum()*100


# fig, ax = plt.subplots()
# ax.bar(df.index, df["value"], color="C0")
# ax2 = ax.twinx()
# ax2.plot(df.index, df["cumpercentage"], color="C1", marker="D", ms=7)
# ax2.yaxis.set_major_formatter(PercentFormatter())

# ax.tick_params(axis="y", colors="C0")
# ax2.tick_params(axis="y", colors="C1")
# plt.suptitle("Парета")


# data = [(abs(key), value) for key, value in od.items()]
# plt.figure(figsize=(7, 6))
# plt.pie([x[0] for x in data], labels=[str(x[0]) for x in data], autopct='%1.1f%%')
# plt.gca().set_aspect('equal')
# plt.suptitle("Кругова")

plt.show()







print(f"Номер варіанту: {variant}\nВідповідно до варіанту n = {n} sigma = {sigma}")
print(f"Математичне сподівання: {calculate_mean(sample_dict).round(4)}")
print(f"Вибіркове середнє:{calculate_mean(sample_dict)}")
print(f"Медіана:{calc_median(sample_dict)}")
print(f"Мода:{calc_mode(sample_dict)}")
print(f"Дисперсія:{calc_variance(sample_dict)}")
print(f"Середньоквадратичне відхилення:{calc_standard_deviation(sample_dict)}")
print_dict(sample_dict)
