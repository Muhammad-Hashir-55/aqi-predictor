t = int(input())

for i in range(t):
    n = int(input())
    arr= list(map(int,input().split()))

    out = [arr[0]]
    for j in range(1,n):
        if(arr[j]<arr[j-1]):
            out.append(1)
            out.append(arr[j])
        else:
            out.append(arr[j])
    print(len(out))
    for j in out:
        print(j,end=' ')
    
