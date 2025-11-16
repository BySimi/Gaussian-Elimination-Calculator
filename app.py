import streamlit as st
import numpy as np

st.title("Gaussian Elimination Calculator")

st.write("Enter number of variables and the augmented matrix values.")

n = st.number_input("Number of variables:", min_value=1, max_value=10, value=3)

matrix_input = st.text_area(
    "Enter the augmented matrix (each row on a new line, values separated by spaces):",
    value="2 1 -1 8\n-3 -1 2 -11\n-2 1 2 -3"
)



def gaussian_elimination(A):
    A = A.astype(float)
    n = len(A)

    for i in range(n):

        
        # pivot = A[i][i]
        # if pivot == 0:
        
        max_row = i + np.argmax(abs(A[i:, i]))
        A[[i, max_row]] = A[[max_row, i]]  # Swap rows



        # A[i] = A[i] / A[i][i]
        for j in range(i+1, n):
            ratio = A[j][i] / A[i][i]

            # A[j][i] = 0
            # for k in range(i+1, n+1):
            #     A[j][k] = A[j][k] - ratio*A[i][k]
            #
            A[j] = A[j] - ratio * A[i]


    x = np.zeros(n)
    for i in range(n-1, -1, -1):

        
        # sum_ax = 0
        # for k in range(i+1, n):
        #     sum_ax += A[i][k] * x[k]
        # x[i] = (A[i][-1] - sum_ax) / A[i][i]
        x[i] = (A[i][-1] - np.dot(A[i][i+1:n], x[i+1:n])) / A[i][i]

    return x



if st.button("Calculate"):
    try:
        
        rows = matrix_input.strip().split("\n")
        A = np.array([list(map(float, row.split())) for row in rows])

        
        if A.shape != (n, n+1):
            st.error(f"Matrix must be of size {n} x {n+1}.")
        else:
            solution = gaussian_elimination(A.copy())
            st.success("Solution:")

            for i, val in enumerate(solution, start=1):
                st.write(f"x{i} = {val:.4f}")

    except Exception as e:
        st.error(f"Invalid input! Error: {e}")



#     coeff = A[:, :-1]
#     const = A[:, -1]
#     sol2 = np.linalg.solve(coeff, const)
#     st.write("Alternate method:", sol2)
