
'''
包含线路参数：站间长度、位置离散、计划运行时间、线路限速、坡度、曲率、实际运行能耗
其中section1-6为上行，7-12为下行
'''

class Section1:
    def __init__(self):
        self.start_station = '''JiuJiangBei'''
        self.end_station = '''MingGuang'''
        self.length = 6640
        self.scheduled_time = 275.78
        self.station_stop_time = 20
        self.speed_limit = {
            0: 45, 334: 110, 1088: 100, 3306: 140, 6197: 80, 6640: 0
        }
        self.gradient = {
            0: 0, 412: -24.58, 862: -10.8, 1842: 28.5, 2422: 11, 3262: -7.84, 4012: 9, 4612: 3.5, 6012: 9.74, 6397: 0
        }
        self.curve = {0: 0, 274: 3500, 440: 0, 491: 3500, 641: 0, 1449: 600, 1874: 0, 2150: 600, 3170: 0, 3361: 3500,
                      3567: 0, 3638: 3500, 3844: 0, 4004: 2500, 4298: 0}
        self.direction = "ShangXing"
        self.tra_power = 159.38
        self.re_power = 58.78
        self.ac_power = 100.6


class Section2:
    def __init__(self):
        self.start_station = '''MingGuang'''
        self.end_station = '''WenQuanDaDao'''
        self.scheduled_time = 104.144
        self.speed_limit = {
            0: 80, 197: 110, 1509: 80, 1980: 0
        }
        self.gradient = {
            0: 0, 171: -20, 421: -3.5, 921: 10, 1471: 22, 1801: 0
        }
        self.curve = {0: 0, 1449: 5000, 1610: 0, 1721: 1200, 1876: 0}
        self.direction = "ShangXing"
        self.tra_power = 56.22
        self.re_power = 22.62
        self.ac_power = 33.6


class Section3:
    def __init__(self):
        self.start_station = '''WenQuanDaDao'''
        self.end_station = '''FengXiHe'''
        self.length = 2040
        self.scheduled_time = 104.177
        self.speed_limit = {
            0: 80, 206: 120, 1583: 80, 2040: 0
        }
        self.gradient = {
            0: 0, 247: -2, 527: -3, 1912: 0
        }
        self.curve = {0: 0, 1250: 6000, 1369: 0, 1758: 6000, 1879: 0}
        self.direction = "ShangXing"
        self.tra_power = 68.14
        self.re_power = 40.68
        self.ac_power = 27.46


class Section4:
    def __init__(self):
        self.start_station = '''FengXiHe'''
        self.end_station = '''ShiWuYiYuan'''
        self.length = 1850
        self.scheduled_time = 105.129
        self.speed_limit = {
            0: 80, 197: 100, 1120: 70, 1850: 0
        }
        self.gradient = {
            0: 0, 144: 10, 1689: 0
        }
        self.curve = {0: 0, 235: 800, 631: 0, 873: 3000, 1155: 0, 1462: 450, 1779: 0}
        self.direction = "ShangXing"
        self.tra_power = 63
        self.re_power = 21.96
        self.ac_power = 41.04


class Section5:
    def __init__(self):
        self.start_station = '''ShiWuYiYuan'''
        self.end_station = '''HuangShi'''
        self.length = 2960
        self.scheduled_time = 138.319
        self.speed_limit = {
            0: 95, 925: 120, 2520: 80, 2960: 0
        }
        self.gradient = {
            0: 0, 248: -25, 648: 17, 1508: 3.25, 2808: 29, 2788: 0
        }
        self.curve = {0: 0, 100: 600, 382: 0, 474: 600, 852: 0, 1012: 1200, 1651: 0, 1847: 3500, 2042: 0, 2650: 5000,
                      2723: 0, 2763: 5000, 2838: 0}
        self.direction = "ShangXing"
        self.tra_power = 101.8
        self.re_power = 28.7
        self.ac_power = 73.1


class Section6:
    def __init__(self):
        self.start_station = '''HuangShi'''
        self.end_station = '''JinXing'''
        self.length = 4280
        self.scheduled_time = 172.1
        self.speed_limit = {
            0: 100, 760: 140, 3832: 80, 4280: 0
        }
        self.gradient = {
            0: 0, 255: 5.9, 2095: 0, 2995: 8.97, 3945: 0
        }
        self.curve = {0: 0, 287: 600, 606: 0, 818: 1004, 1146: 0, 2565: 5004, 2705: 0, 4027: 1000, 4149: 0}
        self.direction = "ShangXing"
        self.tra_power = 119.62
        self.re_power = 46.62
        self.ac_power = 73



class Section7:
    def __init__(self):
        self.start_station = '''MingGuang'''
        self.end_station = '''JiuJiangBei'''
        self.length = 6640
        self.scheduled_time = 258.642
        self.speed_limit = {
            0: 80, 220: 140, 3014: 100, 4232: 95, 5098: 120, 6187: 80, 6640: 0
        }
        self.gradient = {
            0: 0, 244: -9.74, 629: -3.5, 2029: -9.0, 2629: 7.84, 3379: -11.0, 4219: -28.5, 4799: 10.8, 5779: 24.58,
            6229: 0
        }
        self.curve = {0: 0, 2343: 2500, 2637: 0, 2797: 3500, 3003: 0, 3074: 3500, 3280: 0, 3471: 600, 4491: 0,
                      4767: 600,
                      5192: 0, 6000: 3500, 6150: 0, 6201: 3500, 6367: 0}
        self.direction = "XiaXing"
        self.tra_power = 142.4
        self.re_power = 61.76
        self.ac_power = 80.64


class Section8:
    def __init__(self):
        self.start_station = '''WenQuanDaDao'''
        self.end_station = '''MingGuang'''
        self.length = 1974
        self.scheduled_time = 101.081
        self.speed_limit = {
            0: 80, 244: 120, 1530: 80, 1980: 0
        }
        self.gradient = {
            0: 0, 173: -22, 503: -10, 1053: 3.5, 1553: 20, 1803: 0
        }
        self.curve = {0: 0, 98: 1200, 253: 0, 364: 5000, 525: 0}
        self.direction = "XiaXing"
        self.tra_power = 61.04
        self.re_power = 16.88
        self.ac_power = 44.16


class Section9:
    def __init__(self):
        self.start_station = '''FengXiHe'''
        self.end_station = '''WenQuanDaDao'''
        self.length = 2040
        self.scheduled_time = 103.309
        self.speed_limit = {
            0: 80, 235: 120, 1597: 80, 2040: 0
        }
        self.gradient = {
            0: 0, 126: 3, 1511: 2, 1791: 0
        }
        self.curve = {0: 0, 159: 6000, 280: 0, 669: 6000, 788: 0}
        self.direction = "XiaXing"
        self.tra_power = 67.26
        self.re_power = 32.1
        self.ac_power = 35.16


class Section10:
    def __init__(self):
        self.start_station = '''ShiWuYiYuan'''
        self.end_station = '''FengXiHe'''
        self.length = 1850
        self.scheduled_time = 106.156
        self.speed_limit = {
            0: 70, 431: 100, 1445: 80, 1850: 0
        }
        self.gradient = {
            0: 0, 162: -10, 1707: 0
        }
        self.curve = {0: 0, 72: 450, 389: 0, 696: 3000, 978: 0, 1220: 800, 1616: 0}
        self.direction = "XiaXing"
        self.tra_power = 48.38
        self.re_power = 34.16
        self.ac_power = 14.22


class Section11:
    def __init__(self):
        self.start_station = '''HuangShi'''
        self.end_station = '''ShiWuYiYuan'''
        self.length = 2960
        self.scheduled_time = 137.973
        self.speed_limit = {
            0: 100, 337: 120, 1721: 95, 2960: 0
        }
        self.gradient = {
            0: 0, 175: -29, 1155: -3.25, 1455: -17, 2315: 25, 2715: 0
        }
        self.curve = {0: 0, 125: 5000, 200: 0, 240: 5000, 313: 0, 921: 3500, 1116: 0, 1312: 1200, 1621: 0, 1673: 1200,
                      1951: 0, 2111: 600, 2489: 0, 2581: 600, 2863: 0}
        self.direction = "XiaXing"
        self.tra_power = 55.34
        self.re_power = 45.04
        self.ac_power = 10.3


class Section12:
    def __init__(self):
        self.start_station = '''JinXing'''
        self.end_station = '''HuangShi'''
        self.length = 4280
        self.scheduled_time = 171.15
        self.speed_limit = {
            0: 100, 547: 140, 2958: 115, 3298: 100, 4280: 0
        }
        self.gradient = {
            0: 0, 335: -8.97, 1285: 0, 2185: -5.9, 4025: 0
        }
        self.curve = {0: 0, 131: 1000, 253: 0, 1575: 5004, 1715: 0, 3134: 1004, 3462: 0, 3674: 600, 3993: 0}
        self.direction = "XiaXing"
        self.tra_power = 92.94
        self.re_power = 58.64
        self.ac_power = 34.3


Section = {"Section1": Section1(), "Section2": Section2(), "Section3": Section3(), "Section4": Section4(),
           "Section5": Section5(), "Section6": Section6(), "Section7": Section7(), "Section8": Section8(),
           "Section9": Section9(), "Section10": Section10(), "Section11": Section11(), "Section12": Section12()}
