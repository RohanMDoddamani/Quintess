test_list = {"gfg": 2, "is" : 3, "best" : 4, "CS" : 9}
j = {i:test_list[i] for i in sorted(list(test_list.keys()))}
print(j)

list1 = [1,2,3,4,5,6]
print(list1[::-1])
list2 = [i*i for i in list1 if i%2 == 0]
print(list2)
list3 = map(lambda x:x%2 == 0,list)
print(list(list3))


t = (1,2,3,4,5)
print(t[::-1])

