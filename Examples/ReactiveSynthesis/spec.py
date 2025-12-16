import sys

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python spec.py <states> <transitions>")
        sys.exit(1)

    S = int(sys.argv[1])  # number of states
    T = int(sys.argv[2])  # number of transitions

    def same_controller(A,B):
        stmts = []
        for t in range(T):
            stmts.append(f"(from_{t}[{A}] = from_{t}[{B}] & to_{t}[{A}] = to_{t}[{B}] & waiting_{t}_left[{A}] = waiting_{t}_left[{B}] & waiting_{t}_right[{A}] = waiting_{t}_right[{B}] & green_{t}_left[{A}] = green_{t}_left[{B}] & green_{t}_right[{A}] = green_{t}_right[{B}])")
        return " & ".join(stmts)

    def same_waiting(A,B):
        stmts = []
        stmts.append(f"(waiting_left[{A}] = waiting_left[{B}] & waiting_right[{A}] = waiting_right[{B}])")
        return "G (" + " & ".join(stmts) + ")"
    
    def assumption(A):
        return f"G (waiting_left[{A}] -> X !waiting_left[{A}]) & G (waiting_right[{A}] -> X !waiting_right[{A}])"
    
    def guarantee(A):
        return f"G (!green_left[{A}] | !green_right[{A}]) & G (waiting_left[{A}] -> F green_left[{A}]) & G (waiting_right[{A}] -> F green_right[{A}])"

    print("Forall A . Exists B . Forall C . !(" + same_waiting("B","C") + " & " + same_controller("A","C") + " & (" +  assumption("B") + " -> "  + guarantee("C") + "))")