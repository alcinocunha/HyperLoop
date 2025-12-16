import sys
import itertools

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python system.py <states> <transitions>")
        sys.exit(1)

    S = int(sys.argv[1])  # number of states
    T = int(sys.argv[2])  # number of transitions

    print("MODULE main")
    print("FROZENVAR")

    for t in range(T):
        print(f"    from_{t} : 0..{S-1};")
        print(f"    to_{t} : 0..{S-1};")
        print(f"    waiting_{t}_left : boolean;")
        print(f"    waiting_{t}_right : boolean;")
        print(f"    green_{t}_left : boolean;")
        print(f"    green_{t}_right : boolean;")

    print("VAR")
    print(f"    state : 0..{S-1};")
    print(f"    waiting_left : boolean;")
    print(f"    waiting_right : boolean;")
    print(f"    green_left : boolean;")
    print(f"    green_right : boolean;")

    print("INIT")
    print("    state = 0")

    print("TRANS")

    def transition(t):
        stmts = []
        stmts.append(f"state = from_{t} & next(state) = to_{t}")
        stmts.append(f"waiting_{t}_left = waiting_left & waiting_{t}_right = waiting_right")
        stmts.append(f"green_left = green_{t}_left & green_right = green_{t}_right")
        return " & ".join(stmts)

    def no_transition():
        stmts = []
        for t in range(T):
            stmts.append(f"(from_{t} = state -> (waiting_{t}_left != waiting_left | waiting_{t}_right != waiting_right))")
        stmts.append("!green_left & !green_right")
        stmts.append("next(state) = state")
        return " & ".join(stmts)

    print("    " + " |\n    ".join([transition(t) for t in range(T)] + [no_transition()]))