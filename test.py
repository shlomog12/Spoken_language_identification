


# d = {}
# li = ['a','b','c','a','a','b','d']
# for x in li:
#     d[x] = d.get(x, 0) + 1
#
# print(d)
switcher = {}
def get_num_by_tag(tag):
    switcher[tag] = switcher.get(tag, len(switcher))
    return switcher[tag]

li = ['a','b','c','a','a','b','d']
nums = [0] * len(li)
for x in range(len(li)):
    nums[x] = get_num_by_tag(li[x])

print(nums)


# for i in range(1000):
#     for j in range(100):
#         if j > 20:
#             break
#         print(f'{i}   - {j}')