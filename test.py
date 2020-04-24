def fact(num):
    if num < 3:
        return num
    else:
        return num * fact(num -1)

print(fact(250))