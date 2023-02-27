import sys

if __name__ == "__main__":
    student_id = sys.argv[1] # This will be passed by the driver process of TA.
    student_file = __import__(student_id + "_hw0") # Import your code.
    ans_obj = student_file.Solution()
    ans_obj.solve() # The output to stdout will be received by the driver process.