import matplotlib.pyplot as plt
from matplotlib import style


style.use("ggplot")

model_name = "model -1589867511"
#copy the model name paste it here , from previous program or just copy the model number from model_graph.log

def create_acc_loss_graph(model_name):
	contents = open("model_graph.log","r").read().split('\n')

	times = []
	accuracies = []
	losses = []

	val_accs = []
	val_losses = []

	for c in contents:
		if model_name in c:
			name , timestamp , acc , loss , val_acc , val_loss = c.split(",")

			times.append(float(timestamp))
			accuracies.append(float(acc))
			losses.append(float(loss))

			val_accs.append(float(val_acc))
			val_losses.append(float(val_loss))
#getting values of time , accuracy and loss in a list
	fig = plt.figure()

	ax1 = plt.subplot2grid((2,1), (0,0))
	ax2 = plt.subplot2grid((2,1), (1,0), sharex=ax1)
#plotting two graphs accuracy vs time and loss vs time

	ax1.plot(times, accuracies, label="acc")
	ax1.plot(times, val_accs, label="val_acc")
	ax1.legend(loc=2)
	ax2.plot(times,losses, label="loss")
	ax2.plot(times,val_losses, label="val_loss")
	ax2.legend(loc=2)
	plt.show()


create_acc_loss_graph(model_name)