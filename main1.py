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
    # Розширення матриці А вектором b
    Ab = np.hstack((A, b.reshape(-1, 1)))
    print_augmented(Ab)
    vars = np.arange(n)  # Стежимо за перестановками стовпців для фінального порядку змінних
    swap_count = 0
    max_list = []  # Стежимо за максимума для обрахування визначника
    
    # Прямий порядок
    for k in range(n):
        # Пошук максимального елемента у підматриці Ab[k:n, k:n]
        print("\nКрок", k + 1, "\n")
        print_augmented(Ab)
        max_val = -1
        imax, jmax = k, k
        for i in range(k, n):
            for j in range(k, n):
                if abs(Ab[i, j]) > max_val:
                    max_val = abs(Ab[i, j])
                    imax, jmax = i, j
        
        if max_val == 0:
            raise ValueError("Matrix is singular")
        
        print("Максимальний за модулем елемент:", max_val)

        max_list.append(max_val)
        
        # Переставляємо рядки за необхідності
        if imax != k:
            Ab[[k, imax]] = Ab[[imax, k]]
            swap_count += 1
        
        # Переставляємо стовпчики за необхідності
        if jmax != k:
            Ab[:, [k, jmax]] = Ab[:, [jmax, k]]
            vars[[k, jmax]] = vars[[jmax, k]]
            swap_count += 1

        print("Перестановка рядків та стовпчиків:")
        print_augmented(Ab)

        # Матриця для зведення до трикутного вигляду
        M = np.eye(n)
        M[k, k] = 1 / Ab[k, k]
        for i in range(k + 1, n):
            M[i, k] = -Ab[i, k] / Ab[k, k]
        
        # Множимо М на розширену матрицю
        Ab = M @ Ab
        print("Матриця в кінці кроку:")
        print_augmented(Ab)

    # Зворотній хід
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = Ab[i, n]
        for j in range(i + 1, n):
            x[i] -= Ab[i, j] * x[j]
    
    # Враховуємо перестановки стовпчиків
    x_final = np.zeros(n)
    x_final[vars] = x

    # Обраховуємо визначник 
    determinant = (-1) ** swap_count
    for max_val in max_list:
        determinant *= max_val

    return x_final, determinant

def print_augmented(Ab):
    print("Матриця А:\n", Ab[:, :n])
    print("Вектор b:\n", Ab[:, n])

# 3. Метод Зейделя
def seidel_method(A, b, eps, max_iterations=1000):
    n = len(b)
    x = np.zeros_like(b)  # Початкове наближення (вектор нулів)
    
    for iteration in range(max_iterations):
        x_new = np.copy(x)
        
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])  # Використовуємо нові значення з поточної ітерації
            sum2 = np.dot(A[i, i+1:], x[i+1:])  # Використовуємо старі значення з попередньої ітерації
            
            x_new[i] = round((b[i] - sum1 - sum2) / A[i, i], 2)
        
        # Перевіряємо умову припинення
        if np.linalg.norm(x_new - x, ord=np.inf) < eps:
            return x_new, iteration+1
        
        x = x_new
    
    raise ValueError("Метод Зейделя не збігся за задану кількість ітерацій")

# 4. Обчислення числа обумовленості
def condition_number(A):
    A_inv = np.linalg.inv(A)

    norm_A = np.linalg.norm(A[:, :], ord=np.inf)
    norm_A_inv = np.linalg.norm(A_inv[:, :], ord=np.inf)

    print("Норма А:", norm_A)
    print("Норма А^(-1):", norm_A_inv)

    return norm_A * norm_A_inv

# Приклад використання:
n = 3
A, b = get_dummy_gauss()
#A, b = get_dummy_seidel()
#A, b = generate_dd_matrix_and_vector(n)

print("Матриця А:\n", A)
print("Вектор b:\n", b)

# Метод Гауса з вибором головного елемента по всій матриці
print("\nМетод Гаусса з вибором головного по всій матриці\n")
A_gauss = A.copy()
b_gauss = b.copy()
x_gauss, determinant = gauss_elimination_full_pivoting(A_gauss, b_gauss)
print("Розв'язок методом Гауса з вибором головного елемента по всій матриці:\n", x_gauss)

# Визначник
#det_A = determinant_full_pivoting(A_gauss, swap_count)
print("Визначник матриці А:\n", determinant)

# Метод Зейделя
print("\nМетод Зейделя:\n")
A_seidel = A.copy()
b_seidel = b.copy()
eps = 0.5
x_seidel, iterations = seidel_method(A_seidel, b_seidel, eps)
print("Розв'язок методом Зейделя:\n", x_seidel)
print("Кількість ітерацій:\n", iterations)

# Число обумовленості

print("\nЧисло обумовленості А:\n", condition_number(A))
