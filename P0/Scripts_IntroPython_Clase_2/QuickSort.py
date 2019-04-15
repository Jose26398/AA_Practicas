# -*- coding: utf-8 -*-
def someGreatFunction(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    print('pivot:', pivot)
    left = [x for x in arr if x < pivot]
    print('left: ', left)
    middle = [x for x in arr if x == pivot]
    print('middle: ', middle)
    right = [x for x in arr if x > pivot]
    print('right: ', right)
    print('-----------------------------------------')
    return someGreatFunction(left) + middle + someGreatFunction(right)


print(someGreatFunction([3,6,8,10,1,2,1]))


