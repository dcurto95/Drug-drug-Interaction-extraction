import matplotlib.pyplot as plt

false_pairs = {2: 2311, 4: 1397, 6: 1124, 5: 967, 7: 928, 8: 922, 3: 911, 10: 796, 9: 771, 12: 720, 11: 693, 14: 623, 13: 575,
               15: 490, 16: 487, 17: 476, 18: 463, 20: 400, 19: 398, 21: 346, 22: 333, 24: 290, 23: 281, 26: 250, 25: 235,
               28: 219, 30: 192, 27: 190, 32: 184, 29: 163, 34: 144, 31: 132, 37: 112, 35: 111, 36: 111, 33: 111, 38: 104,
               42: 100, 1: 95, 39: 95, 44: 95, 46: 93, 40: 92, 41: 86, 48: 77, 43: 65, 49: 65, 45: 64, 0: 63, 52: 59, 50: 57,
               60: 55, 54: 54, 56: 52, 53: 51, 51: 51, 58: 50, 70: 43, 63: 42, 47: 41, 55: 41, 62: 39, 66: 37, 72: 37, 64: 37,
               67: 34, 74: 34, 65: 34, 57: 33, 59: 33, 76: 32, 68: 31, 61: 30, 69: 26, 78: 23, 81: 22, 79: 21, 80: 21, 71: 20,
               73: 18, 77: 18, 85: 18, 82: 18, 84: 18, 83: 17, 86: 17, 75: 16, 87: 16, 88: 14, 90: 14, 93: 14, 89: 14, 91: 12,
               95: 11, 97: 11, 99: 10, 92: 10, 96: 9, 104: 9, 101: 8, 94: 8, 98: 8, 106: 8, 107: 7, 118: 7, 102: 7, 100: 7,
               103: 7, 110: 6, 116: 6, 109: 6, 111: 6, 113: 6, 120: 6, 105: 6, 108: 6, 112: 5, 114: 5, 130: 5, 125: 5, 123: 4,
               128: 4, 115: 4, 122: 4, 135: 3, 121: 2, 133: 2, 127: 2, 132: 2, 117: 2, 124: 2, 119: 1, 138: 1, 139: 1, 145: 1,
               136: 1, 142: 1, 134: 1, 140: 1, 131: 1, 137: 1, 129: 1, 126: 1}
true_pairs = {2: 321, 7: 232, 6: 199, 8: 193, 5: 182, 11: 180, 9: 175, 4: 160, 3: 157, 10: 156, 12: 150, 14: 125, 13: 114,
              15: 94, 16: 93, 18: 84, 17: 72, 19: 71, 20: 62, 23: 55, 21: 52, 22: 49, 24: 41, 27: 40, 25: 37, 26: 32, 29: 28,
              30: 26, 28: 25, 35: 22, 33: 20, 32: 19, 31: 18, 34: 13, 38: 13, 37: 12, 36: 11, 40: 10, 42: 10, 39: 10, 41: 9,
              45: 8, 0: 7, 49: 7, 44: 7, 48: 7, 47: 6, 43: 6, 46: 5, 51: 5, 54: 4, 58: 4, 65: 3, 75: 3, 52: 3, 87: 3, 89: 3,
              93: 3, 97: 3, 111: 3, 53: 2, 57: 2, 60: 2, 62: 2, 72: 2, 61: 2, 82: 2, 85: 2, 91: 2, 100: 2, 102: 2, 109: 2,
              113: 2, 56: 2, 67: 2, 70: 2, 95: 2, 50: 2, 64: 1, 66: 1, 71: 1, 73: 1, 81: 1, 105: 1, 107: 1, 119: 1, 122: 1,
              124: 1, 126: 1, 128: 1, 131: 1, 133: 1, 135: 1, 140: 1, 142: 1, 145: 1, 147: 1, 150: 1, 151: 1, 157: 1, 69: 1,
              55: 1, 59: 1, 63: 1, 77: 1, 84: 1, 99: 1, 101: 1, 103: 1, 108: 1, 68: 1, 74: 1, 76: 1, 78: 1, 80: 1, 83: 1,
              117: 1}
plt.figure(figsize=(20, 20))

p1 = plt.bar(false_pairs.keys(), false_pairs.values())
p2 = plt.bar(true_pairs.keys(), true_pairs.values())

plt.ylabel('Count')
plt.xlabel('Entity distance')
plt.title('Counts for entity distance')
plt.legend((p1[0], p2[0]), ('False pairs', 'True pairs'))

plt.show()
