import sys
import itertools

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python controller.py <states> <transitions>")
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
    print("INIT")
    constraints = []
    for t1, t2 in itertools.combinations(range(T), 2):
            constraints.append(f"(from_{t1} = from_{t2} -> (waiting_{t1}_left != waiting_{t2}_left | waiting_{t1}_right != waiting_{t2}_right))")
    print("    " + " &\n    ".join(constraints))
