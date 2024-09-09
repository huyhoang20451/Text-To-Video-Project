import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

math_df = pd.read_csv('math_datasets_with_labels.csv')

# Clean the data
#Lấy mẫu dữ liệu 100% dữ liệu gốc
df_sample = math_df.sample(frac=1, random_state=42)

#Tiền xử lý dữ liệu đơn giản
import re
def clean_text(text):
    # Loại bỏ các ký tự đặc biệt và chuyển văn bản thành chữ thường
    text = re.sub(r'[^\w\s]','', text)
    text = text.lower()
    return text

#Áp dụng làm sạch dữ liệu
df_sample['Input Text'] = df_sample['Input Text'].apply(clean_text)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000) # Giới hạn số lượng đặc trưng để giảm tài nguyên
X = vectorizer.fit_transform(df_sample['Input Text'])
y = df_sample['Label']

#Khi lượng dữ liệu nhỏ, làm sao để tối ưu hóa được mô hình
from sklearn.model_selection import train_test_split, cross_val_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Huấn luyện mô hình
from sklearn.linear_model import LogisticRegression
#Support Vector Machine (SVM)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
#KNN
from sklearn.neighbors import KNeighborsClassifier
#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#Logistic Regression
model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

#Random Forest
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

#SVM
model_svm = SVC()
model_svm.fit(X_train, y_train)
#KNN
model_knn = KNeighborsClassifier()
model_knn.fit(X_train, y_train)

#Gradient Boosting
model_gb = GradientBoostingClassifier()
model_gb.fit(X_train, y_train)

#So sánh hiệu suất của 5 mô hình trên
models = ['Logistic Regression', 'Random Forest', 'SVM', 'KNN', 'Gradient Boosting']
accuracies = [accuracy_score(y_test, model.predict(X_test)) for model in [model_lr, model_rf, model_svm, model_knn, model_gb]]
plt.bar(models, accuracies)
plt.ylabel('Accuracy')
plt.show()

#Vẽ ma trận nhầm lẫn cho cả 5 mô hình
from sklearn.metrics import confusion_matrix
import seaborn as sns

fig, axes = plt.subplots(3, 2, figsize=(20, 20))
for i, model in enumerate([model_lr, model_rf, model_svm, model_knn, model_gb]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    sns.heatmap(cm, annot=True, ax=axes[i//2, i%2], fmt='d', cmap='Blues')
    axes[i//2, i%2].set_title(models[i])
plt.show()

from gtts import gTTS

def text_to_speech(text, output_file, lang='en'):
    if not isinstance(text, str):
        text = str(text)  # Chuyển đổi thành chuỗi nếu không phải chuỗi
    tts = gTTS(text, lang=lang)
    tts.save(output_file)
    return output_file

import re
import sympy as sp

# Hàm giải phương trình bậc 1 với các bước chi tiết
def solve_linear_equation_step_by_step(input_text):
    # Tách và làm sạch phương trình từ văn bản đầu vào
    equation_match = re.search(r'Solve for x:\s+(.*)', input_text)
    if not equation_match:
        return "Can't find the equation in the input text.", None  # Return two values: result and audio_file
    equation = equation_match.group(1).strip()
    print(f'The function input: {equation}')
    
    # Xử lý các trường hợp thiếu dấu nhân giữa số và biến (2x -> 2*x)
    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
    
    # Tách phương trình thành hai phần (trái và phải dấu "=")
    lhs, rhs = equation.split('=')
    
    # Tạo biểu thức phương trình bằng SymPy
    expr = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
    
    # Giải phương trình
    x = sp.symbols('x')
    solution = sp.solve(expr, x)
    
    # Khởi tạo danh sách lưu các bước giải
    steps = []
    
    # Bước 1: Chuyển vế phải của phương trình sang trái (chuyển tất cả về một vế)
    steps.append(f"Step 1: {lhs} - ({rhs}) = 0")
    
    # Tạo một biểu thức tạm thời
    temp_expr = sp.sympify(lhs) - sp.sympify(rhs)
    
    # Bước 2: Đơn giản hóa phương trình
    simplified_expr = sp.simplify(temp_expr)
    steps.append(f"Step 2: {simplified_expr} = 0")
    
    # Bước 3: Giải phương trình để tìm x
    if len(solution) == 1:
        steps.append(f"Step 3: x = {solution[0]}")
    else:
        steps.append(f"Step 3: {simplified_expr} = 0 -> No solution or multiple solutions")
    
    steps_text = "\n".join(steps)

    # Kết quả cuối cùng
    result_text = f"The solution of this equation is: x = {solution[0]}\n\n" \
                  f"Step-by-step to solve the function:\n{steps_text}"

    # In ra kết quả cuối cùng
    print(f"The solution of this equation is: x = {solution[0]}")
    print("Step-by-step to solve the function:")
    for step in steps:
        print(step)
    
    # Giả lập hàm chuyển văn bản thành giọng nói và trả về đường dẫn file âm thanh
    audio_file = text_to_speech(result_text, 'solve_linear_equation.mp3')
    
    # Trả về kết quả và file âm thanh
    return result_text, audio_file

import re
import sympy as sp

#Giải phương trình bậc hai với các bước chi tiết
def solve_quadratic_equation_step_by_step(input_text):
    # Tìm kiếm phương trình bậc hai trong văn bản đầu vào
    equation_match = re.search(r'quadratic equation.*?:\s*(.*)', input_text, re.IGNORECASE)
    if not equation_match:
        return "Can't find the equation in the input text."
    
    equation = equation_match.group(1).strip()
    print(f'The function input: {equation}')
    
    # Xử lý các trường hợp thiếu dấu nhân giữa số và biến (2x -> 2*x)
    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
    
    # Tách phương trình thành hai phần (trái và phải dấu "=")
    lhs, rhs = equation.split('=')
    
    # Tạo biểu thức phương trình bằng SymPy
    x = sp.symbols('x')
    expr = sp.Eq(sp.sympify(lhs), sp.sympify(rhs))
    
    # Đưa phương trình về dạng chuẩn ax^2 + bx + c = 0
    lhs_expr = sp.sympify(lhs)
    rhs_expr = sp.sympify(rhs)
    simplified_expr = sp.simplify(lhs_expr - rhs_expr)
    
    # Tính toán các hệ số a, b, c
    coeffs = sp.Poly(simplified_expr, x).all_coeffs()
    a, b, c = map(lambda coef: float(coef), coeffs) if len(coeffs) == 3 else (0, 0, 0)

    # Tính toán delta
    delta = b**2 - 4*a*c
    steps = []
    
    # Bước 1: Chuyển vế phải của phương trình sang trái (chuyển tất cả về một vế)
    steps.append(f"Step 1: {lhs} - ({rhs}) = 0")
    
    # Bước 2: Đơn giản hóa phương trình
    steps.append(f"Step 2: {simplified_expr} = 0")
    
    # Bước 3: Tính delta
    steps.append(f"Step 3: Calculate delta = b^2 - 4ac = {b}^2 - 4*{a}*{c} = {delta}")
    
    # Bước 4: Xét giá trị của delta
    if delta > 0:
        # Phương trình có hai nghiệm phân biệt
        sqrt_delta = sp.sqrt(delta)
        x1 = (-b + sqrt_delta) / (2 * a)
        x2 = (-b - sqrt_delta) / (2 * a)
        steps.append(f"Step 4: Delta > 0, the equation has two distinct solutions:")
        steps.append(f"x1 = (-{b} + sqrt({delta})) / (2*{a}) = {x1}")
        steps.append(f"x2 = (-{b} - sqrt({delta})) / (2*{a}) = {x2}")
    elif delta == 0:
        # Phương trình có nghiệm kép
        x1 = -b / (2 * a)
        steps.append(f"Step 4: Delta = 0, the equation has a double root")
        steps.append(f"x1 = x2 = -{b} / (2*{a}) = {x1}")
    else:
        # Phương trình vô nghiệm thực
        steps.append(f"Step 4: Delta < 0, the equation has no real roots")
    
    steps_text = "\n".join(steps)

    result_text = f"The solutions of this equation are: \n" 
    if delta >= 0:
        if delta > 0:
            result_text += f"x1 = {x1}\n"
            result_text += f"x2 = {x2}\n\n"
        else:
            result_text += f"x1 = x2 = {x1}\n\n"
    else:
        result_text += "The equation has no real roots.\n\n"
    result_text += f"Step-by-step to solve the function:\n{steps_text}"

    #Add audio file
    audio_file = text_to_speech(result_text, 'solve_quadratic_equation.mp3')
    
    # In ra kết quả cuối cùng
    print(result_text)
    return result_text, audio_file

import sympy as sp
import re

def solve_systems_of_equations_v2(text_input):
    # Tìm các phương trình trong văn bản đầu vào
    equations_match = re.search(r'Solve the system of equations: (.*), (.*)', text_input)
    if not equations_match:
        return "Can't find the equations in the input text.", None  # Return two values

    eq1 = equations_match.group(1).strip()
    eq2 = equations_match.group(2).strip()

    # Tách phương trình thành hai vế
    x, y = sp.symbols('x y')
    eq1_lhs, eq1_rhs = eq1.split('=')
    eq2_lhs, eq2_rhs = eq2.split('=')

    # Chuyển đổi chuỗi thành biểu thức SymPy
    eq1_expr = sp.Eq(sp.parse_expr(eq1_lhs.strip()), sp.parse_expr(eq1_rhs.strip()))
    eq2_expr = sp.Eq(sp.parse_expr(eq2_lhs.strip()), sp.parse_expr(eq2_rhs.strip()))

    # Giải hệ phương trình
    system = [eq1_expr, eq2_expr]
    solution = sp.linsolve(system, x, y)

    # Cung cấp chi tiết từng bước dưới dạng LaTeX
    steps = []

    # Thay thế biểu thức trong hệ phương trình
    steps.append(f'Equation 1: \\({sp.latex(eq1_expr)}\\)')
    steps.append(f'Equation 2: \\({sp.latex(eq2_expr)}\\)')

    # Giải hệ phương trình
    steps.append(f'\\(\\text{{Solve the system of equations: }} {sp.latex(eq1_expr)}, {sp.latex(eq2_expr)}\\)')

    # Chuyển hệ phương trình về dạng ma trận
    A, B = sp.linear_eq_to_matrix(system, (x, y))
    steps.append(f'Solve the system of equations by matrix method:')
    steps.append(f"\\(A = \\begin{{matrix}}{sp.latex(A)}\\end{{matrix}}\\)")
    steps.append(f"\\(B = \\begin{{matrix}}{sp.latex(B)}\\end{{matrix}}\\)")

    # Tính toán và giải hệ phương trình
    steps.append(f'Solve the system of equations:')
    sol = sp.linsolve((A, B), x, y)
    steps.append(f'The values of system of equations are: \\({sp.latex(sol)}\\)')

    text_results = "\n".join(steps)
    audio_file = text_to_speech(text_results, 'solve_systems_of_equations.mp3')

    return text_results, audio_file  # Ensure two values are returned

import re
import sympy as sp

def solve_derivative_step_by_step(input_text):
    # Tách và làm sạch biểu thức từ văn bản đầu vào, loại bỏ 'f(x)='
    equation_match = re.search(r'Find the derivative of\s*f\(x\)=\s*(.*)', input_text)
    if not equation_match:
        return "Can't find the equation in the input text."
    
    expression = equation_match.group(1).strip()
    print(f'The equation input: {expression}')
    
    # Xử lý các trường hợp thiếu dấu nhân giữa số và biến (2x -> 2*x)
    expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
    
    # Tạo biểu thức bằng SymPy
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    
    # Tính đạo hàm
    derivative = sp.diff(expr, x)
    
    # Khởi tạo danh sách lưu các bước giải
    steps = []
    
    # Bước 1: Hiển thị biểu thức ban đầu
    steps.append(f"Step 1: The equation input: {expr}")
    # Bước 2: Tính đạo hàm
    steps.append(f"Step 2: Calculate the derivative of the equation")
    steps.append(f"Step 3: Derivatives = {derivative}")
    
    # In ra kết quả cuối cùng
    print("Result:")
    print(f"The derivative expression of {expression} is {derivative}")

    print("\nStep-by-step to solve the derivative:")
    for step in steps:
        print(step)
    
    steps_text = "\n".join(steps)
    result_text = f"The derivative expression of {expression} is {derivative}\n\n" \
                  f"Step-by-step to solve the derivative:\n{steps_text}"
    
    audio_file = text_to_speech(result_text, 'math_solution.mp3', lang='en')

    return result_text, audio_file


#Hàm giải tích phân
def solve_integral_step_by_step(input_text):
    # Tách và làm sạch biểu thức từ văn bản đầu vào, loại bỏ 'f(x)='
    equation_match = re.search(r'Find the integral of\s*f\(x\)=\s*(.*)', input_text)
    if not equation_match:
        return "Can't find the equation in the input text."
    
    expression = equation_match.group(1).strip()
    print(f'The equation input: {expression}')
    
    # Xử lý các trường hợp thiếu dấu nhân giữa số và biến (2x -> 2*x)
    expression = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expression)
    
    # Tạo biểu thức bằng SymPy
    x = sp.symbols('x')
    expr = sp.sympify(expression)
    
    # Tính tích phân
    integral = sp.integrate(expr, x)
    
    # Khởi tạo danh sách lưu các bước giải
    steps = []
    
    # Bước 1: Hiển thị biểu thức ban đầu
    steps.append(f"Step 1: The equation input: {expr}")
    # Bước 2: Tính tích phân
    steps.append(f"Step 2: Calculate the integral of the equation with respect to x")
    steps.append(f"Step 3: Integral = {integral}")
    
    # In ra kết quả cuối cùng
    print("Result:")
    print(f"The inrtegral of {expression} is {integral}")
    
    print("\nStep-by-step to solve the integral:")
    for step in steps:
        print(step)
    
    steps_text = "\n".join(steps)
    result_text = f"The integral of {expression} is {integral}\n\n" \
                  f"Step-by-step to solve the integral:\n{steps_text}"
    
    audio_file = text_to_speech(result_text, 'math_solution.mp3', lang='en')

    return result_text, audio_file


import re
import sympy as sp

def solve_matrix_determinant_step_by_step(input_text):
    # Tách và làm sạch ma trận từ văn bản đầu vào
    matrix_match = re.search(r'matrix\s+(\[\[.*\]\])', input_text)
    if not matrix_match:
        return "Can't find the equation in the input text."
    
    matrix_str = matrix_match.group(1).strip()
    print(f'The matrix input: {matrix_str}')
    
    # Tạo ma trận từ văn bản
    try:
        # Chuyển chuỗi ma trận thành danh sách các danh sách
        matrix_list = sp.Matrix(eval(matrix_str))
    except Exception as e:
        return f"Error occurred when execute matrix: {e}"
    
    # Tính định thức
    determinant = matrix_list.det()
    
    # Khởi tạo danh sách lưu các bước giải
    steps = []
    
    # Bước 1: Hiển thị ma trận ban đầu
    steps.append(f"Step 1: The matrix input: {matrix_list}")
    
    # Bước 2: Xác định công thức tính định thức
    if matrix_list.shape == (2, 2):
        a, b, c, d = matrix_list[0, 0], matrix_list[0, 1], matrix_list[1, 0], matrix_list[1, 1]
        formula = f"Determinant = a*d - b*c = {a}*{d} - {b}*{c}"
        determinant_value = a*d - b*c
        steps.append(f"Step 2: Apply the formula: Determinant = a*d - b*c with 2x2 matrix")
        steps.append(f"  Formula: Determinant = a*d - b*c")
        steps.append(f"  With a = {a}, b = {b}, c = {c}, d = {d}")
        steps.append(f"  Apply: {formula} = {determinant_value}")
    else:
        steps.append(f"Step 2: Calculate the determinant of the matrix.")
    
    # Bước 3: Hiển thị kết quả
    steps.append(f"Step 3: Determinant = {determinant}")
    
    # In ra kết quả cuối cùng
    print("Result:")
    print(f"The determinants of this matrix is: {determinant}")
    
    print("\nStep-by-step to solve the determinant:")
    for step in steps:
        print(step)
    
    steps_text = "\n".join(steps)
    result_text = f"The determinants of this matrix is: {determinant}\n\n" \
                  f"Step-by-step to solve the determinant:\n{steps_text}"
    
    audio_file = text_to_speech(result_text, 'math_solution.mp3', lang='en')
    
    return result_text, audio_file


#Hàm tính eigenvalues
def solve_matrix_eigenvalues_step_by_step(input_text):
    # Tách và làm sạch ma trận từ văn bản đầu vào
    matrix_match = re.search(r'matrix\s+(\[\[.*\]\])', input_text)
    if not matrix_match:
        return "Can't find the equation in the input text."
    
    matrix_str = matrix_match.group(1).strip()
    print(f'The matrix input: {matrix_str}')
    
    # Tạo ma trận từ văn bản
    try:
        # Chuyển chuỗi ma trận thành danh sách các danh sách
        matrix_list = sp.Matrix(eval(matrix_str))
    except Exception as e:
        return f"Error occured when execute matrix: {e}"
    
    # Tính eigenvalues
    eigenvalues = matrix_list.eigenvals()
    
    # Khởi tạo danh sách lưu các bước giải
    steps = []
    
    # Bước 1: Hiển thị ma trận ban đầu
    steps.append(f"Step 1: The matrix input: {matrix_list}")
    
    # Bước 2: Tính eigenvalues
    steps.append(f"Step 2: Calculate the eigenvalues of the matrix.")
    
    # Bước 3: Hiển thị kết quả
    steps.append(f"Step 3: Eigenvalues = {eigenvalues}")
    
    # In ra kết quả cuối cùng
    print("Result:")
    print(f"Eigenvalues of the matrix is: {eigenvalues}")
    
    print("\nStep-by-step to solve the eigenvalues:")
    for step in steps:
        print(step)
    
    steps_text = "\n".join(steps)
    result_text = f"Eigenvalues of the matrix is: {eigenvalues}\n\n" \
                  f"Step-by-step to solve the eigenvalues:\n{steps_text}"
    
    audio_file = text_to_speech(result_text, 'math_solution.mp3', lang='en')

    return result_text, audio_file


def solve_math_problem(text_input):
    # Vectorize the input text to identify the problem type
    vectorized_text = vectorizer.transform([text_input])

    # Predict the type of math problem
    problem_type = model_knn.predict(vectorized_text)[0]

    # Solve the problem based on its type
    if problem_type == 0:
        # Solve a linear equation to find x
        result, audio_file = solve_linear_equation_step_by_step(text_input)
        return result, audio_file
    elif problem_type == 1:
        # Solve an integral
        result, audio_file = solve_quadratic_equation_step_by_step(text_input)
        return result, audio_file
    elif problem_type == 2:
        # Calculate system of equations
        result, audio_file = solve_systems_of_equations_v2(text_input)
        return result, audio_file
    elif problem_type == 3:
        # Calculate derivative of the function
        result, audio_file = solve_derivative_step_by_step(text_input)
        return result, audio_file
    elif problem_type == 4:
        # Calculate the integral of the function
        result, audio_file = solve_integral_step_by_step(text_input)
        return result, audio_file
    elif problem_type == 5:
        # Calculate the determinant of the matrix
        result, audio_file = solve_matrix_determinant_step_by_step(text_input)
        return result, audio_file
    elif problem_type == 6:
        # Calculate the eigenvalues of the matrix
        result, audio_file = solve_matrix_eigenvalues_step_by_step(text_input)
        return result, audio_file
    else:
        # If no problem type is matched
        return "Can't solve this math problem.", None


from manim import *
# from sympy import Matrix, symbols, solve, Eq, latex

# class MathProblem(Scene):
#     def construct(self):
#         # Set up the background color
#         self.camera.background_color = BLACK
        
#         # Create a title with smaller text
#         title = Text('Solving Math Problem', font='Arial', stroke_width=0).scale(1.0)  # Reduced scale for smaller text
#         self.play(Write(title), run_time=1)
#         self.wait(1)
        
#         # Create a math problem description with smaller text
#         text_input = 'Calculate the eigenvalues of matrix [[1, 7], [3, 4]]'
#         math_problem = Text(text_input, font='Arial', stroke_width=0).scale(0.6)  # Smaller scale
#         math_problem.next_to(title, DOWN, buff=0.5)  # Add spacing (buff) to avoid overlapping
#         self.play(Write(math_problem), run_time=1)
#         self.wait(1)
        
#         #Fade out the title and math problem
#         self.play(FadeOut(title), FadeOut(math_problem))
#         self.wait(1) # Wait for a moment after fade-out

#         # Display matrix in LaTeX format with smaller text
#         matrix_tex = MathTex(r"\text{Matrix input: } \begin{bmatrix} 1 & 7 \\ 3 & 4 \end{bmatrix}").scale(0.6)
#         matrix_tex.next_to(math_problem, DOWN, buff=0.5)  # Add spacing (buff) to avoid overlapping
#         self.play(Write(matrix_tex), run_time=1)
#         self.wait(1)

#         #Push the matrix input at the top center of the screen
#         matrix_tex.to_edge(UP)

#         # Solve the math problem step-by-step
#         steps = self.solve_eigenvalues_latex()
        
#         # Use VGroup to manage vertical spacing and alignment
#         steps_group = VGroup()
#         for i, step in enumerate(steps):
#             step_tex = MathTex(step).scale(0.6)  # Display each step in smaller LaTeX format
#             if i == 0:
#                 step_tex.next_to(matrix_tex, DOWN, buff=0.5)  # Start below matrix_tex
#             else:
#                 step_tex.next_to(steps_group[-1], DOWN, buff=0.3)  # Position relative to previous step
#             steps_group.add(step_tex)  # Add each step to the VGroup
#             self.play(Write(step_tex), run_time=1)
#             self.wait(1)

#     def solve_eigenvalues_latex(self):
#         # Define the matrix
#         matrix = Matrix([[1, 7], [3, 4]])
        
#         # Step-by-step solution in LaTeX format with reduced text size
#         step1 = r"\text{Step 1: Matrix input: } \begin{bmatrix} 1 & 7 \\ 3 & 4 \end{bmatrix}"
        
#         # Calculate eigenvalues
#         eigenvals = matrix.eigenvals()
#         step2 = r"\text{Step 2: Calculate the eigenvalues of the matrix.}"
        
#         # Display the result in LaTeX format
#         solution_text = r"\text{Eigenvalues: } " + ', '.join([latex(val) for val in eigenvals.keys()])
#         step3 = rf"Step\ 3: {solution_text}"
        
#         # Return the LaTeX-formatted steps
#         return [step1, step2, step3]

#Voi bat ky dang Toan nao
from manim import *

class MathProblem(Scene):
    def construct(self):
        # Nhận đầu vào văn bản và giải bài toán
        text_input = input('Enter a math problem:')
        result, audio_file = solve_math_problem(text_input)
        
        # Hiển thị bài toán
        problem_text = Text(text_input, font='Arial').scale(0.6)
        self.play(Write(problem_text))
        self.wait(1)
        
        # Fade out bài toán
        self.play(FadeOut(problem_text))

        # Hiển thị các bước giải toán
        steps = self.get_steps_from_result(result)
        steps_group = VGroup()
        for i, step in enumerate(steps):
            step_tex = MathTex(step).scale(0.6)
            if i == 0:
                step_tex.next_to(problem_text, DOWN, buff=0.5)
            else:
                step_tex.next_to(steps_group[-1], DOWN, buff=0.5)
            steps_group.add(step_tex)
            self.play(Write(step_tex))
            self.wait(1)
        
        # Hiển thị kết quả
        result_text = Text(result, font='Arial').scale(0.5)
        #Voi moi buoc giai, hien thi len sau do an di sau 1s
        self.play(Write(result_text))
        self.wait(1)
        self.play(FadeOut(result_text))
        self.wait(1)
        result_text.next_to(steps_group, DOWN, buff=0.5)

        self.play(Write(result_text))
        self.wait(2)
        
        # Phát âm thanh nếu có
        if audio_file:
            self.add_sound(audio_file, time_offset=0)

    def get_steps_from_result(self, result):
        # Tách các bước giải toán từ kết quả (giả định kết quả là danh sách các bước)
        return result.split('\n')

# Save this script as `math_problem.py` and run:
# `manim -pql math_problem.py MathProblem`


if __name__ == "__main__":
    result_video = MathProblem()
    result_video.render()


