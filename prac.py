n = int(input())

for i in range(n):
    arr = list(map(int,input().split()))
    s = set(arr)
    if(len(s) ==1):
        print('YES')
    else:
        print('NO')