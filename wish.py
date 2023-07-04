import random
import inspect
import sys

up_four_stars = ["烦恼刈除·久岐忍(雷)", "梦园藏金·多莉(雷)", "绮思晚星·莱依拉(冰)"]
four_stars = ["烦恼刈除·久岐忍(雷)", "梦园藏金·多莉(雷)", "绮思晚星·莱依拉(冰)", ]
up_five_stars = ["白草净华·纳西妲(草)"]
five_stars = ["白草净华·纳西妲(草)"]

"""
原神的每次祈愿，分三步:
1.确定稀有度（三星/四星/五星)
2.确定类型(UP/歪，角色/武器)
3.确定具体内容(从既定范围内选择一项)

"""
p_character_five_stars = 0.6
p_weapon_five_stars = 0.7
p_character_four_stars = 5.1
p_weapon_four_stars = 6


# 角色up池
def update_character_wish_information(result, num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost,
                                      four_is_lost):
    if result == 0:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = num_of_four_stars_wish + 1

    elif result == 1:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = 1
        four_is_lost = 1

    elif result == 2:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = 1
        four_is_lost = 1

    elif result == 3:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = 1
        four_is_lost = 0

    elif result == 4:
        num_of_five_stars_wish = 1
        num_of_four_stars_wish = num_of_four_stars_wish + 1
        five_is_lost = 1

    elif result == 5:

        num_of_five_stars_wish = 1
        num_of_four_stars_wish = num_of_four_stars_wish + 1
        five_is_lost = 0

    updated_wish_information = [num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost]
    return updated_wish_information


def character_wish(num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost):
    """
    0, 三星
    1，四星武器
    2，四星常驻角色
    3，四星up角色
    4，五星常驻角色
    5，五星up角色
    :param num_of_five_stars_wish: 当前五星抽数
    :param num_of_four_stars_wish: 当前四星抽数
    :param is_lost: 是否已经歪
    :return: 抽卡结果
    """
    # 根据抽数，确定概率
    # 四星概率
    # print(num_of_four_stars_wish)
    if num_of_four_stars_wish <= 8:
        p_four_stars = p_character_four_stars
    elif num_of_four_stars_wish == 9:
        p_four_stars = 56.1
    else:
        p_four_stars = 100

    # 五星概率
    if num_of_five_stars_wish <= 73:
        p_five_stars = p_character_five_stars
    elif num_of_five_stars_wish >= 74 and num_of_five_stars_wish < 90:
        p_five_stars = p_character_five_stars + (num_of_five_stars_wish - 73) * 6
    else:
        p_five_stars = 100

    # 确定稀有度
    location = random.random() * 100
    # print(f"[{__file__} {sys._getframe().f_lineno}] -> 四星抽数：{num_of_four_stars_wish} 四星概率：{p_four_stars}，五星抽数：{num_of_five_stars_wish}，五星概率：{p_five_stars}")

    if location <= p_five_stars:  # 五星
        if five_is_lost == 1:
            result = 5
        else:
            population = [4, 5]
            weights = [0.5, 0.5]
            result = random.choices(population, weights=weights, k=1)[0]
    elif location < (p_five_stars + p_four_stars):  # 四星
        if four_is_lost == 1:
            result = 3
        else:
            population = [1, 2, 3]
            weights = [0.25, 0.25, 0.5]
            result = random.choices(population, weights=weights, k=1)[0]
    else:  # 三星
        result = 0

    num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost = update_character_wish_information(
        result,
        num_of_five_stars_wish,
        num_of_four_stars_wish,
        five_is_lost,
        four_is_lost)
    return result, num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost


def character_ten_wish(num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost):
    results = []
    for i in range(0, 10):
        result, num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost = character_wish(
            num_of_five_stars_wish=num_of_five_stars_wish,
            num_of_four_stars_wish=num_of_four_stars_wish,
            five_is_lost=five_is_lost,
            four_is_lost=four_is_lost)
        results.append(result)
        # print(num_of_four_stars_wish)
    return results, [num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost]


# 武器up池
def update_weapon_wish_information(result, num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost):
    if result == 0:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = num_of_four_stars_wish + 1

    elif result == 1:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = 1
        four_is_lost = 1

    elif result == 2:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = 1
        four_is_lost = 1

    elif result == 3:
        num_of_five_stars_wish = num_of_five_stars_wish + 1
        num_of_four_stars_wish = 1
        four_is_lost = 0

    elif result == 4:
        num_of_five_stars_wish = 1
        num_of_four_stars_wish = num_of_four_stars_wish + 1
        five_is_lost = 1

    elif result == 5:

        num_of_five_stars_wish = 1
        num_of_four_stars_wish = num_of_four_stars_wish + 1
        five_is_lost = 0

    updated_wish_information = [num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost]
    return updated_wish_information


def weapon_wish(num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost):
    """
    0, 三星
    1，四星武器
    2，四星常驻角色
    3，四星up武器
    4，五星常驻武器
    5，五星up武器
    :param num_of_five_stars_wish: 当前五星抽数
    :param num_of_four_stars_wish: 当前四星抽数
    :param is_lost: 是否已经歪
    :return: 抽卡结果
    """
    # 根据抽数，确定概率
    # 五星概率
    if num_of_five_stars_wish <= 63:
        p_five_stars = p_weapon_five_stars
    elif num_of_five_stars_wish <= 73:
        p_five_stars = p_weapon_five_stars + (num_of_five_stars_wish - 62) * 7
    elif num_of_five_stars_wish <= 79:
        p_five_stars = 77.7 + (num_of_five_stars_wish - 73) * 3.5
    else:
        p_five_stars = 100

    # 四星概率
    # print(num_of_four_stars_wish)
    if num_of_four_stars_wish <= 7:
        p_four_stars = p_weapon_four_stars
    elif num_of_four_stars_wish == 8:
        p_four_stars = 66
    elif num_of_four_stars_wish == 9:
        p_four_stars = 96
    else:
        p_four_stars = 100

    # 确定稀有度
    location = random.random() * 100
    # print(f"[{__file__} {sys._getframe().f_lineno}] -> 四星抽数：{num_of_four_stars_wish} 四星概率：{p_four_stars}，五星抽数：{num_of_five_stars_wish}，五星概率：{p_five_stars}")

    if location <= p_five_stars:  # 五星
        # if num_of_five_stars_wish > 74:
        #     print(f"[{sys._getframe().f_lineno}] -> 第{num_of_five_stars_wish}抽出五星")
        if five_is_lost == 1:
            result = 5
        else:
            population = [4, 5]
            weights = [0.25, 0.75]
            result = random.choices(population, weights=weights, k=1)[0]

    elif location < (p_five_stars + p_four_stars):  # 四星
        if four_is_lost == 1:
            result = 3
        else:
            population = [1, 2, 3]
            weights = [0.25, 0.25, 0.50]
            result = random.choices(population, weights=weights, k=1)[0]
    else:  # 三星
        result = 0

    # 更新抽卡信息
    num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost = update_weapon_wish_information(
        result,
        num_of_five_stars_wish,
        num_of_four_stars_wish,
        five_is_lost,
        four_is_lost)
    return result, num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost


def weapon_ten_wish(num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost):
    results = []
    for i in range(0, 10):
        result, num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost = weapon_wish(
            num_of_five_stars_wish=num_of_five_stars_wish,
            num_of_four_stars_wish=num_of_four_stars_wish,
            five_is_lost=five_is_lost,
            four_is_lost=four_is_lost)
        results.append(result)
        # print(num_of_four_stars_wish)
    return results, [num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost]


if __name__ == '__main__':
    num_of_five_stars_wish = 1
    num_of_four_stars_wish = 1
    five_is_lost = 0
    four_is_lost = 0

    wish_information = [num_of_five_stars_wish, num_of_four_stars_wish, five_is_lost, four_is_lost]

    """
    0, 三星
    1，四星武器
    2，四星常驻角色
    3，四星up角色
    4，五星常驻角色
    5，五星up角色
    """
    n = 0
    five_star = 0
    character_banner = ["三星", "四星常驻武器", "四星常驻角色", "四星up角色", "五星常驻角色", "五星up角色"]
    weapon_banner = ["三星", "四星常驻武器", "四星常驻角色", "四星up武器", "五星常驻角色", "五星up角色"]

    result_list = []
    # 测试十连多金的次数
    while (True):
        result, wish_information = character_ten_wish(*wish_information)
        n = n + 1
        five_star = 0
        for i in result:
            result_list = []
            if i == 4 or i == 5:
                five_star = five_star + 1
        result_list = []
        if five_star > 0:
            for item in result:
                result_list.append(character_banner[item])
            # print(result_list)
        if five_star >= 2:
            print(f"第{n}次十连出了双黄")
            break


    # while (five_star < 100000):
    #     result, wish_information = weapon_ten_wish(*wish_information)
    #     for i in result:
    #         if i == 4 or i == 5:
    #             five_star = five_star + 1

    # for item in result:
    #     if item == 0:
    #         print("三星")
    #         num_of_five_stars_wish = num_of_five_stars_wish + 1
    #         num_of_four_stars_wish = num_of_four_stars_wish + 1
    #
    #
    #     elif item == 1:
    #         print("四星武器")
    #         num_of_five_stars_wish = num_of_five_stars_wish + 1
    #         num_of_four_stars_wish = 1
    #         four_is_lost = 1
    #     elif item == 2:
    #         print("四星常驻角色")
    #         num_of_five_stars_wish = num_of_five_stars_wish + 1
    #         num_of_four_stars_wish = 1
    #         four_is_lost = 1
    #
    #     elif item == 3:
    #         print("四星up角色")
    #         num_of_five_stars_wish = num_of_five_stars_wish + 1
    #         num_of_four_stars_wish = 1
    #         four_is_lost = 0
    #
    #     elif item == 4:
    #         print("五星常驻角色")
    #         num_of_five_stars_wish = 1
    #         num_of_four_stars_wish = num_of_four_stars_wish + 1
    #         five_is_lost = 1
    #     elif item == 5:
    #         print("五星up角色")
    #         num_of_five_stars_wish = 1
    #         num_of_four_stars_wish = num_of_four_stars_wish + 1
    #         five_is_lost = 0
