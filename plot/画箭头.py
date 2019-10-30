import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3,3,100)
y = x**2

x_l = [-3,-2]
y_l = [9,4]
position_text = {'qwe':(-3,-9),'rty':(-2,4)}   # 定义的是箭头初始的位置和箭头需要指示的内容

plt.axis([-3,3,0,10])

for name,pos_text in position_text.items():
    plt.annotate(name,xy=(1,1),xytext=pos_text,arrowprops=dict(facecolor='black',width=0.5,shrink=0.1,headwidth=5))
plt.plot(x,y,'r')
plt.show()

# name:注释的内容
#xy:设置所要标注的位置坐标
#xytext:设置注释内容显示的起始位置
# arrowprops 用来设置箭头
# facecolor 设置箭头的颜色
# headlength 箭头的头的长度
# headwidth 箭头的宽度
# width 箭身的宽度