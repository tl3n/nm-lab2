import numpy as np

def get_dummy_gauss():
    A = np.array([[1, 2, 3],
               [2, 5, 5],
               [3, 5, 6]], dtype=float)
    b = np.array([1, 2, 3], dtype=float)

    return A, b

def get_dummy_seidel():
    A = np.array([[3, -1, 1],
              [-1, 2, 0.5],
              [1, 0.5, 3]], dtype=float)

    b = np.array([1, 1.75, 2.5], dtype=float)

    return A, b

# 1. Генерація діагонально домінуючої матриці та вектора
def generate_dd_matrix_and_vector(n):
    A = np.random.rand(n, n) * 10  # Генерація випадкової матриці n x n
    b = np.random.rand(n) * 10  # Генерація вектора правої частини
    
    # Робимо матрицю діагонально домінуючою
    for i in range(n):
        A[i, i] = sum(np.abs(A[i])) + np.random.rand() * 10  # Домінуючий елемент на діагоналі
    
    return A, b

# 2. Метод Гауса з вибором головного елемента по всій матриці
def gauss_elimination_full_pivoting(A, b):
    n = len(b)
    # Індекси для відстеження перестановок стовпців
    index = np.arange(n)
    # Кількість перестановок
    swap_count = 0

    # Прямий хід
    for k in range(n):
        # Знаходження максимального елемента по всій підматриці A[k:n, k:n]
        max_row, max_col = np.unravel_index(np.argmax(np.abs(A[k:, k:])), A[k:, k:].shape)
        max_row += k
        max_col += k

        if A[max_row, max_col] == 0:
            raise ValueError("Matrix is singular")

        # Перестановка рядків
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
            swap_count += 1

        # Перестановка стовпців
        if max_col != k:
            A[:, [k, max_col]] = A[:, [max_col, k]]
            index[[k, max_col]] = index[[max_col, k]]
            swap_count += 1

        # Виконання елімінації
        for i in range(k+1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотний хід для знаходження розв'язку
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]

    # Враховуємо перестановку стовпців
    x_final = np.zeros(n)
    x_final[index] = x

    return x_final, swap_count

# 3. Метод Зейделя
def seidel_method(A, b, eps, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b)  # Початкове наближення (вектор нулів)
    
    for iteration in range(max_iterations):
        x_new = np.copy(x)
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # Використовуємо нові значення з поточної ітерації
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # Використовуємо старі значення з попередньої ітерації
            
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        
        # Перевіряємо умову припинення
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, iteration+1
        
        x = x_new
    
    raise ValueError("Метод Зейделя не збігся за задану кількість ітерацій")

# 4. Обчислення визначника методом Гауса з вибором головного елемента по всій матриці
def determinant_full_pivoting(A, swap_count):
    det = np.prod(np.diag(A)) * (-1)**swap_count
    return det

# 5. Обчислення числа обумовленості
def condition_number(A):
    eigenvalues = np.linalg.eigvals(A)
    return max(abs(eigenvalues)) / min(abs(eigenvalues))

# Приклад використання:
n = 4
#A, b = get_dummy_gauss()
#A, b = get_dummy_seidel()
A, b = generate_dd_matrix_and_vector(n)

print("Matrix A (diagonally dominant):\n", A)
print("Vector b:\n", b)

# Метод Гауса з вибором головного елемента по всій матриці
A_gauss = A.copy()
b_gauss = b.copy()
x_gauss, swap_count = gauss_elimination_full_pivoting(A_gauss, b_gauss)
print("Solution using Gauss elimination with full pivoting:\n", x_gauss)
print("Number of swaps:\n", swap_count)

# Метод Зейделя
A_seidel = A.copy()
b_seidel = b.copy()
eps = 1e-10
x_seidel, iterations = seidel_method(A_seidel, b_seidel, eps)
print("Solution using Seidel method:\n", x_seidel)
print("Number of iterations:\n", iterations)

# Визначник
det_A = determinant_full_pivoting(A_gauss, swap_count)
print("Determinant of A:\n", det_A)

# Число обумовленості
cond_num = condition_number(A)
print("Condition number of A:\n", cond_num)
