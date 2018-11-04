from subprocess import Popen,PIPE
import matplotlib.pyplot as plt
import random

def terminal(command):
	result = Popen(args=command,stdout = PIPE,shell=True)
	result.wait()
	result=result.stdout.read().decode("utf-8") 
	return result


def plot(mtype,x,y):
	title='CPU Execution time for a '+mtype+' matrix'
	plt.title(title)
	plt.xlabel('Size of Matrix')
	plt.ylabel('Inversion time (s)')
	style=''
	if mtype=='identity':
		style='kx'
	if mtype=='hollow':
		style='bo'
	if mtype=='band':
		style='c^'
	if mtype=='sparse':
		style='yh'
	if mtype=='dense':
		style='r*'
	plt.plot(x, y, style)
	plt.savefig(mtype+'.png')
	plt.show()
	plt.close()

types=['dense','sparse','band','hollow','identity']

results={}
for mtype in types:
	results[mtype]={'x':[],'y':[]}


for mtype in types:
	print('Type of matrix =',mtype)
	for i in range(50,1050,50):
		results[mtype]['x'].append(i)
		time=float(terminal(['.\\cpu.exe','input/'+mtype+'/'+str(i)+'.txt',str(i)]))*1000
		print('Matrix of size ',i,' completed in ',time)
		results[mtype]['y'].append(time)
	#plot(mtype,results[mtype]['x'],results[mtype]['y'])
	print()


