import random
import numpy

def generate_dense_matrix(n):
	"""Function generates a dense matrix"""
	matrix=[]
	for i in range(n):
		row=[]
		for j in range(n):
			row.append(random.randint(0,2**12))
		matrix.append(row)
	return matrix

def generate_sparse_matrix(n):
	"""Function generates a sparse matrix"""
	matrix=generate_dense_matrix(n)
	for i in range(random.randint(0,n)):
		index=random.randint(0,n-1)
		matrix[index][index]=0
	return matrix


def generate_identity_matrix(n):
	"""Function generates a identity matrix"""
	matrix=[]
	for i in range(n):
		row=[]
		for j in range(n):
			if i==j:
				row.append(1)
			else:
				row.append(0)
		matrix.append(row)
	return matrix

def generate_band_matrix(n):
	"""Function generates a band matrix"""
	band_width=random.randint(2,n-1)
	matrix=[]
	for i in range(n):
		row=[]
		for j in range(n):
			in_band=False
			for k in range(band_width):
				if i==j+k or j==i+k:
					in_band=True
			if in_band:
				row.append(random.randint(0,2**12))
			else:
				row.append(0)

		matrix.append(row)
	return matrix

def generate_hollow_matrix(n):
	"""Function generates a hollow matrix"""
	matrix=[]
	for i in range(n):
		row=[]
		for j in range(n):
			if i==j:
				row.append(0)
			else:
				row.append(random.randint(0,2**12))
		matrix.append(row)
	return matrix


def print_matrix(matrix):
	for row in matrix:
		for col in row:
			print(col,end=' ')
		print('\n') 

def write_to(directory,output,n):
	with open('input/'+directory+'/'+str(n)+'.txt', 'w') as f:
		for i in range(len(output)):
			for j in range(len(output[i])):
				if j==len(output[i])-1:
					f.write("%s" % str(output[i][j]))
				else:
					f.write("%s," % str(output[i][j]))
			f.write('\n')
	f.close()

def save_matrix(function,n):
	output = function(n)
	numpy_matrix=numpy.asmatrix(output, dtype='float')
	(sign, det) = numpy.linalg.slogdet(numpy_matrix)
	det=sign * numpy.exp(det)
	while det==0:
		output = function(n)
		numpy_matrix=numpy.asmatrix(output)	
		(sign, det) = numpy.linalg.slogdet(numpy_matrix)
		det=sign * np.exp(det)
	
	if 'identity' in function.__doc__:
		write_to('identity',output,n)
	if 'hollow' in function.__doc__:
		write_to('hollow',output,n)
	if 'dense' in function.__doc__:
		write_to('dense',output,n)
	if 'sparse' in function.__doc__:
		write_to('sparse',output,n)
	if 'band' in function.__doc__:
		write_to('band',output,n)

for i in range(50,1050,50):
	print('n=',i,' matrix generated')
	save_matrix(generate_identity_matrix,i)
	save_matrix(generate_hollow_matrix,i)
	save_matrix(generate_dense_matrix,i)
	save_matrix(generate_sparse_matrix,i)
	save_matrix(generate_band_matrix,i)

