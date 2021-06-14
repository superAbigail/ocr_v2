ori_str = '@19:10(Pre)-21:44(舒张试验）'
str_list = ori_str[1:].split('-')
for s in str_list:
    print(s.split('(')[0])