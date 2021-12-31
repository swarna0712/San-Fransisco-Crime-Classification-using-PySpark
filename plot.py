import matplotlib.pyplot as plt

f1=open('MLP1.txt','r')
x1=f1.read().split("\n")[1:-1]
f1.close()
f1=open('MLP2.txt','r')
x2=f1.read().split("\n")[1:-1]
f1.close()
f1=open('MLP3.txt','r')
x3=f1.read().split("\n")[1:-1]
f1.close()
f1=open('MLP4.txt','r')
x4=f1.read().split("\n")[1:-1]
f1.close()

x1=list(map(lambda x:float(x),x1))
x2=list(map(lambda x:float(x),x2))
x3=list(map(lambda x:float(x),x3))
x4=list(map(lambda x:float(x),x4))

plt.plot(list(range(1,30)),x1,label='1')
plt.plot(list(range(1,30)),x2,label='2')
plt.plot(list(range(1,29)),x3,label='3')
plt.plot(list(range(1,29)),x4,label='4')
# naming the x axis
plt.xlabel('batches')
# naming the y axis
plt.ylabel('accuracy')
# giving a title to my graph
plt.title('MLP Hyper Parameter Tuning')
 
# show a legend on the plot
plt.legend()
plt.show()