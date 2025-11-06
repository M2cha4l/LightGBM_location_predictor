import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(8, 8))
pic_list = ['canyin','fengjing','ggss','gonsiqiye','gouwu','jiaotong','jinrong'
    ,'shenghuo','tiyu','tuoyu','wenhuafuwu','yiliao'
            ,'zhengfu','zhusu','zhuzhai']
k = 0
for i in range(3):
    for j in range(5):
        img = plt.imread(f'./figure/shanghai_poi_kernel_density/1/{pic_list[k]}.png')
        axs[i][j].imshow(img)
        plt.xlim(0,50)
        plt.ylim(0,60)
        plt.title(f'{pic_list[k]}核密度')
        print(pic_list[k])
        k+=1
fig.suptitle('上海市各类核密度')
plt.show()