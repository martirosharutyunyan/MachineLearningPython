import numpy as np
from numba import prange,jit,njit

# arr = np.array([1,2,3,4,5,6])
# print(arr)
# print(arr.shape)
# print(arr.dtype)
# print(arr.ndim)
# np.int64 // Signed 64 bit integer type
# np.float32 // Standard double precision floating point
# np.complex // Complex number represented by 128 floats
# np.bool // Boolean type storing True and False values
# np.object // Python object type 
# np.string_ // Fixed length string type
# np.unicode_ // Fixed length unicode type

# arr = np.array([1,2,3,4,5,6],dtype=np.float64)
# print(arr)
# print(arr.size)
# print(len(arr))

# arr = np.arange(0,20,1.5)
# print(arr)

# arr = np.linspace(0,20,30)
# arr = np.linspace(0,20,30).reshape(5,6)
# print(arr)

# arr = np.random.random((5))
# print(arr)
# if we need to create random numbers array with our diapason do this
# arr = 10 * np.random.random((5)) -5 # random numbers (-5,5) 
# print(arr)

# arr = np.arange(1,11)
# print(np.sqrt(arr))
# print(np.sin(arr))
# print(np.cos(arr))
# print(np.log(arr)) # logarithm e 
# print(np.exp(arr))

# arr = np.arange(1,11)
# arr2 = np.arange(11,21)
# print(arr)
# print(arr2)
# newArray = arr + arr2
# print(newArray)
# newArray = arr * arr2
# print(newArray)
# newArray = arr ** 2
# print(newArray)
# newArray = arr / arr2
# print(newArray)
# newArray = arr -arr2
# print(newArray)

# arr = np.random.randint(-5,10,10)
# print(arr)
# print(arr.max())
# print(arr.min())
# print(arr.mean()) # mijin tvabanakan
# print(arr.sum()) # gumar
# print(arr.std()) # Стандартное отклонение представляет
# print(np.median(arr))
# arr = np.arange(1,11)
# print(arr < 4) # veradarcnuma masiv kazmvac true falsic vortex paymany bavararuma true vortex che false

# arr = np.arange(1,11)
# print(arr)
# arr = np.insert(arr,0,5)
# print(arr)
# arr = np.sort(arr)
# print(arr)
# arr = np.delete(arr,2)
# print(arr)
# arr2 = np.arange(11,21)
# arr = np.concatenate((arr,arr2))
# print(arr)
# arr = np.array_split(arr,11)
# print(arr)

# arr = np.array([1,-2,3,-4,5])
# print(arr)
# print(arr[2])
# print(arr[0:2])
# print(arr[::-1])
# print(arr[arr<2])
# print(arr[(arr < 2) & (arr > 0)])
# print(arr[(arr > 4) | (arr < 0)])
# arr[1:4] = 0
# print(arr)

# matrix = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
#     [10,11,12],
# ], dtype=np.float64)
# print(matrix)
# print(matrix.shape)
# print(matrix.ndim) # 1d or 2d or 3d other matrix  
# print(matrix.size)

# matrix = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
#     [10,11,12],
# ], dtype=np.float64)
# matrix = matrix.reshape(2,6)
# print(matrix)

# matrix = np.random.random((2,4))
# print(matrix)

# matrix = np.array([
#     [1,2,3],
#     [4,5,6],
#     [7,8,9],
#     [10,11,12],
# ], dtype=np.float64)
# matrix = np.resize(matrix,(2,2)) # i tarberutyun sovorakan reshapi tuyl a talis ktrel 2
# print(matrix)

# matrix = np.arange(0,10).reshape(5,2)
# print(matrix)

# matrix = np.zeros((2,3))
# print(matrix)
# matrix = np.ones((2,3))
# print(matrix)
# matrix = np.eye(5) # ankyunagcov 1 era sharum
# print(matrix)
# matrix = np.full((2,3),9)
# print(matrix)
# matrix = np.empty((2,3))
# print(matrix)

# matrix1 = np.array([
#     [1,2],
#     [3,4]
# ])
# print(matrix1)
# matrix2 = np.array([
#     [5,6],
#     [7,8]
# ])
# print(matrix2)
# print(matrix1 + matrix2)
# print(matrix1 - matrix2)
# print(matrix1 * matrix2)
# print(matrix1 ** matrix2)
# print(matrix1 / matrix2)
# print(matrix1.dot(matrix2)) # skalyarnoe proizvedenie

# matrix = np.arange(1,10).reshape(3,3)
# print(matrix)
# matrix = np.delete(matrix,0,axis=0) # x eri arancq
# print(matrix)
# matrix = np.delete(matrix,0,axis=1) # y eri arancq
# print(matrix)   

# matrix = np.arange(1,10).reshape(3,3)
# print(matrix.max(axis=0))
# print(matrix.max(axis=0).max())

# matrix = np.array([
#     [1,2,3,5],
#     [4,5,6,5],
#     [7,8,9,5],
#     [10,11,12,5],
# ])
# matrix = np.array([
#     [1,3,5],
#     [0,6,5],
#     [4,3,5],
# ])
# print(matrix)
# print(np.linalg.inv(matrix))
# print(np.trace(matrix))
# print(np.linalg.det(matrix)) 
# print(np.linalg.eig(matrix))

# from scipy.stats import mode
# print(mode([1,2,3,1,1]))
# x = lambda a,b: a*b+3
# print(x(3,4))

